use std::sync::Arc;

use anyhow::Result;
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::agent::Agent;
use crate::api::{self, LlmProvider};
use crate::channel::{CommandRegistry, MessageHandler};
use crate::config::AppConfig;
use crate::{db, search, tool};

/// 共用資源，傳遞給各 agent 的 channel 和心跳任務
pub struct SharedResources {
    pub pg: Arc<sqlx::PgPool>,
    pub qdrant: Arc<qdrant_client::Qdrant>,
    pub embedder: Arc<api::gemini::GeminiProvider>,
    pub llm: Arc<api::gemini::GeminiProvider>,
    pub worker: Arc<api::gemini::GeminiProvider>,
    pub brain: Arc<api::gemini::GeminiProvider>,
}

/// 啟動所有 agent 的 channel、心跳等任務，回傳 JoinHandle 列表
pub fn spawn_all(
    agents: Vec<Agent>,
    res: &SharedResources,
    cfg: &AppConfig,
) -> Vec<JoinHandle<Result<()>>> {
    let mut handles = Vec::new();

    let heartbeat_interval = cfg.heartbeat_interval;
    let brain_model = cfg
        .brain
        .as_ref()
        .map(|b| b.model.clone())
        .unwrap_or_else(|| cfg.llm.model.clone());

    for agent in agents {
        let agent_id = agent.id;
        let agent_prompt = agent.prompt;
        let agent_soul = agent.soul;
        let agent_heartbeat = agent.heartbeat;
        let agent_brain_heartbeat = agent.brain_heartbeat;
        let agent_path = agent.path;
        let agent_llm_commands = agent.llm_commands;

        // 啟動 LLM 心跳任務
        if !agent_heartbeat.is_empty() {
            info!(agent_id = %agent_id, interval = heartbeat_interval, "啟動 LLM 心跳任務");
            handles.push(tokio::spawn(run_heartbeat(HeartbeatTask {
                pg: Arc::clone(&res.pg),
                worker: Arc::clone(&res.worker),
                llm: Arc::clone(&res.llm),
                agent_id: agent_id.clone(),
                agent_prompt: agent_prompt.clone(),
                heartbeat_prompt: agent_heartbeat,
                interval_secs: heartbeat_interval,
                llm_model: cfg.llm.model.clone(),
            })));
        }

        // 啟動大腦心跳任務
        if !agent_brain_heartbeat.is_empty() {
            info!(agent_id = %agent_id, interval = heartbeat_interval, "啟動大腦心跳任務");
            handles.push(tokio::spawn(run_heartbeat(HeartbeatTask {
                pg: Arc::clone(&res.pg),
                worker: Arc::clone(&res.worker),
                llm: Arc::clone(&res.brain),
                agent_id: agent_id.clone(),
                agent_prompt: agent_prompt.clone(),
                heartbeat_prompt: agent_brain_heartbeat,
                interval_secs: heartbeat_interval,
                llm_model: brain_model.clone(),
            })));
        }

        // 建立指令註冊表
        let mut cmd_registry = CommandRegistry::new();

        cmd_registry.register(
            "ping",
            "檢查機器人是否在線",
            "/ping",
            Arc::new(|_sender, _args| Box::pin(async { Ok("OK".into()) })),
        );

        // 心跳查看/編輯指令
        if let Some(ref apath) = agent_path {
            register_heartbeat_commands(&mut cmd_registry, apath);
        }

        // LLM 互動指令（來自 cmd.toml）
        for cmd_def in &agent_llm_commands {
            let cmd_prompt = cmd_def.prompt.clone();
            let cmd_tools = cmd_def.tools.clone();
            let llm_for_cmd = Arc::clone(&res.llm);
            let pg_for_cmd = Arc::clone(&res.pg);
            let soul_for_cmd = agent_soul.clone();
            let prompt_for_cmd = agent_prompt.clone();
            let aid_for_cmd = agent_id.clone();

            cmd_registry.register(
                cmd_def.name.clone(),
                cmd_def.description.clone(),
                format!("/{} [補充說明]", cmd_def.name),
                Arc::new(move |_sender, args| {
                    let cmd_prompt = cmd_prompt.clone();
                    let cmd_tools = cmd_tools.clone();
                    let llm = Arc::clone(&llm_for_cmd);
                    let pg = Arc::clone(&pg_for_cmd);
                    let soul = soul_for_cmd.clone();
                    let prompt = prompt_for_cmd.clone();
                    let aid = aid_for_cmd.clone();
                    Box::pin(async move {
                        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %:z");

                        let soul_section = if soul.is_empty() {
                            String::new()
                        } else {
                            format!("## 性格設定\n{soul}\n\n")
                        };

                        let system = format!(
                            "你是一個 AI 助手。\n\n\
                             目前時間：{now}\n\n\
                             {soul_section}\
                             {prompt}"
                        );

                        let user_message = if args.trim().is_empty() {
                            cmd_prompt
                        } else {
                            format!("{cmd_prompt}\n\n{}", args.trim())
                        };

                        let result = if cmd_tools.is_empty() {
                            llm.generate(api::GenerateParams {
                                system: Some(&system),
                                user_message: &user_message,
                                json_mode: false,
                                temperature: 0.7,
                            })
                            .await?
                        } else {
                            let tools = tool::build_registry(&cmd_tools, &pg, &aid);
                            api::chat_with_tools(llm.as_ref(), &system, &user_message, &tools)
                                .await?
                        };

                        Ok(result.text)
                    })
                }),
            );
        }

        // help 最後註冊，help_text() 會包含前面所有指令 + help 自己
        cmd_registry.register(
            "help",
            "顯示可用指令列表",
            "/help",
            Arc::new(|_sender, _args| Box::pin(async { Ok(String::new()) })),
        );
        let help_text = cmd_registry.help_text();
        cmd_registry.register(
            "help",
            "顯示可用指令列表",
            "/help",
            Arc::new(move |_sender, _args| {
                let text = help_text.clone();
                Box::pin(async move { Ok(text) })
            }),
        );

        let commands = Arc::new(cmd_registry);

        for channel in agent.channels {
            let msg_ctx = Arc::new(MessageContext {
                pg: Arc::clone(&res.pg),
                qdrant: Arc::clone(&res.qdrant),
                embedder: Arc::clone(&res.embedder),
                llm: Arc::clone(&res.llm),
                worker: Arc::clone(&res.worker),
                agent_id: agent_id.clone(),
                prompt: agent_prompt.clone(),
                soul: agent_soul.clone(),
                worker_model: cfg.worker.model.clone(),
                llm_model: cfg.llm.model.clone(),
            });

            let handler: MessageHandler = Arc::new(move |msg| {
                let ctx = Arc::clone(&msg_ctx);
                Box::pin(async move { handle_message(&ctx, &msg).await })
            });

            let cmds = Arc::clone(&commands);
            handles.push(tokio::spawn(channel.start(handler, cmds)));
        }
    }

    handles
}

/// 註冊心跳查看/編輯指令
fn register_heartbeat_commands(registry: &mut CommandRegistry, apath: &std::path::Path) {
    let hb_path = apath.join("HEARTBEAT.md");
    registry.register(
        "heartbeat",
        "查看或設定心跳提示詞",
        "/heartbeat [show|set <內容>]",
        Arc::new(move |_sender, args| {
            let hb_path = hb_path.clone();
            Box::pin(async move {
                let subcmd = args.trim();
                if subcmd.is_empty() || subcmd == "show" {
                    let content = tokio::fs::read_to_string(&hb_path)
                        .await
                        .unwrap_or_else(|_| "（無 HEARTBEAT.md）".into());
                    Ok(format!("HEARTBEAT.md:\n\n{content}"))
                } else if let Some(new_content) = subcmd.strip_prefix("set ") {
                    tokio::fs::write(&hb_path, new_content).await?;
                    Ok("HEARTBEAT.md 已更新".into())
                } else {
                    Ok("用法: /heartbeat [show] | /heartbeat set <內容>".into())
                }
            })
        }),
    );

    let bhb_path = apath.join("BRAIN_HEARTBEAT.md");
    registry.register(
        "brain_heartbeat",
        "查看或設定大腦心跳提示詞",
        "/brain_heartbeat [show|set <內容>]",
        Arc::new(move |_sender, args| {
            let bhb_path = bhb_path.clone();
            Box::pin(async move {
                let subcmd = args.trim();
                if subcmd.is_empty() || subcmd == "show" {
                    let content = tokio::fs::read_to_string(&bhb_path)
                        .await
                        .unwrap_or_else(|_| "（無 BRAIN_HEARTBEAT.md）".into());
                    Ok(format!("BRAIN_HEARTBEAT.md:\n\n{content}"))
                } else if let Some(new_content) = subcmd.strip_prefix("set ") {
                    tokio::fs::write(&bhb_path, new_content).await?;
                    Ok("BRAIN_HEARTBEAT.md 已更新".into())
                } else {
                    Ok("用法: /brain_heartbeat [show] | /brain_heartbeat set <內容>".into())
                }
            })
        }),
    );
}

/// 訊息處理所需的共用上下文（避免過多參數）
struct MessageContext {
    pg: Arc<sqlx::PgPool>,
    qdrant: Arc<qdrant_client::Qdrant>,
    embedder: Arc<api::gemini::GeminiProvider>,
    llm: Arc<api::gemini::GeminiProvider>,
    worker: Arc<api::gemini::GeminiProvider>,
    agent_id: String,
    prompt: String,
    soul: String,
    worker_model: String,
    llm_model: String,
}

/// 處理單則訊息（RAG 流程）
async fn handle_message(
    ctx: &MessageContext,
    msg: &crate::channel::IncomingMessage,
) -> Result<String> {
    // 0. 載入備忘錄
    let memo = db::postgres::get_scratchpad(&ctx.pg, &ctx.agent_id)
        .await?
        .unwrap_or_default();

    // 1. 入庫（worker 蒸餾 + embedder 向量化）
    let (_, wk_usage) =
        search::ingest(&ctx.pg, &ctx.qdrant, ctx.embedder.as_ref(), ctx.worker.as_ref(), &ctx.agent_id, &msg.text).await?;

    db::postgres::insert_token_usage(&ctx.pg, &ctx.agent_id, &ctx.worker_model, wk_usage.input_tokens, wk_usage.output_tokens).await?;

    // 2. 混合搜尋取回相關記憶
    let results = search::retrieve(&ctx.pg, &ctx.qdrant, ctx.embedder.as_ref(), &ctx.agent_id, &msg.text, 5).await?;

    // 3. 載入近期歷史（填充至 ~5000 token）
    let recent = db::postgres::get_recent_messages(&ctx.pg, &ctx.agent_id, 50).await?;
    const HISTORY_MAX_TOKENS: i32 = 5000;
    let mut history_parts = Vec::new();
    let mut total_tokens = 0i32;
    for (text, token_count) in &recent {
        if total_tokens + token_count > HISTORY_MAX_TOKENS {
            break;
        }
        history_parts.push(text.as_str());
        total_tokens += token_count;
    }
    history_parts.reverse();
    let history = if history_parts.is_empty() {
        "（無近期歷史）".to_owned()
    } else {
        history_parts.join("\n---\n")
    };

    // 4. 建立工具
    let mut tools = tool::ToolRegistry::new();
    for t in tool::scratchpad::tools(Arc::clone(&ctx.pg), ctx.agent_id.clone()) {
        tools.register(t);
    }
    tools.register(tool::fetch::tool());

    // 5. 組合上下文，用主 LLM 帶工具生成回答
    let context_str: String = results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("[記憶 {}] {}", i + 1, r.original_text))
        .collect::<Vec<_>>()
        .join("\n\n");

    let memo_section = if memo.is_empty() {
        "（空）".to_owned()
    } else {
        memo
    };

    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %:z");

    let soul_section = if ctx.soul.is_empty() {
        String::new()
    } else {
        format!("## 性格設定\n{}\n\n", ctx.soul)
    };

    let system = format!(
        "你是一個具有長期記憶的 AI 助手。\n\n\
         目前時間：{now}\n\n\
         {soul_section}\
         {}\n\n\
         ## 備忘錄 (Scratchpad)\n{memo_section}\n\n\
         ## 相關記憶\n{context_str}\n\n\
         ## 近期對話歷史\n{history}\n\n\
         ## 指示\n\
         請根據備忘錄、歷史記憶和近期對話與使用者的新訊息，給出自然、有幫助的回答。\n\
         如果記憶中沒有相關資訊，就正常回答即可。\n\
         如果有重要的事情需要記住（如使用者偏好、關鍵決定、待辦事項），\
         請使用 save_memo 工具儲存到備忘錄中。",
        ctx.prompt
    );

    let result = api::chat_with_tools(ctx.llm.as_ref(), &system, &msg.text, &tools).await?;

    db::postgres::insert_token_usage(&ctx.pg, &ctx.agent_id, &ctx.llm_model, result.input_tokens, result.output_tokens).await?;

    Ok(result.text)
}

const HEARTBEAT_TRIAGE_PROMPT: &str = "\
You are a triage agent. Given the heartbeat instructions, current time, and scratchpad context, \
decide whether there is a task that needs to be executed RIGHT NOW.\n\
Consider time-sensitive tasks, scheduled actions, and pending items.\n\n\
Reply with ONLY \"yes\" (needs action now) or \"no\" (skip this cycle).";

/// 心跳任務設定
struct HeartbeatTask {
    pg: Arc<sqlx::PgPool>,
    worker: Arc<api::gemini::GeminiProvider>,
    llm: Arc<api::gemini::GeminiProvider>,
    agent_id: String,
    agent_prompt: String,
    heartbeat_prompt: String,
    interval_secs: u64,
    llm_model: String,
}

/// 心跳任務：定期喚醒 worker 判斷是否需要呼叫 LLM 執行任務
async fn run_heartbeat(task: HeartbeatTask) -> Result<()> {
    let interval = std::time::Duration::from_secs(task.interval_secs);

    loop {
        tokio::time::sleep(interval).await;

        info!(agent_id = %task.agent_id, "心跳觸發");

        let memo = db::postgres::get_scratchpad(&task.pg, &task.agent_id)
            .await
            .unwrap_or(None)
            .unwrap_or_default();

        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %:z");

        let context = format!(
            "目前時間：{now}\n\n\
             ## 心跳指示\n{}\n\n\
             ## 備忘錄\n{memo}",
            task.heartbeat_prompt
        );

        let triage_result = task
            .worker
            .generate(api::GenerateParams {
                system: Some(HEARTBEAT_TRIAGE_PROMPT),
                user_message: &context,
                json_mode: false,
                temperature: 0.0,
            })
            .await;

        let needs_action = match triage_result {
            Ok(r) => r.text.trim().to_lowercase().starts_with("yes"),
            Err(e) => {
                error!(agent_id = %task.agent_id, "心跳分流判斷失敗: {e:#}");
                continue;
            }
        };

        if !needs_action {
            info!(agent_id = %task.agent_id, "心跳：無需執行任務");
            continue;
        }

        info!(agent_id = %task.agent_id, "心跳：喚醒 LLM 執行任務");

        let mut tools = tool::ToolRegistry::new();
        for t in tool::scratchpad::tools(Arc::clone(&task.pg), task.agent_id.clone()) {
            tools.register(t);
        }
        tools.register(tool::fetch::tool());

        let system = format!(
            "你是一個具有長期記憶的 AI 助手。\n\n\
             目前時間：{now}\n\n\
             {}\n\n\
             ## 備忘錄 (Scratchpad)\n{memo}\n\n\
             ## 指示\n\
             這是一個定期心跳喚醒。請根據以下心跳指示執行任務。\n\
             如果有重要的事情需要記住，請使用 save_memo 工具儲存到備忘錄中。\n\n\
             ## 心跳指示\n{}",
            task.agent_prompt, task.heartbeat_prompt
        );

        let result = api::chat_with_tools(task.llm.as_ref(), &system, "心跳觸發，請執行任務。", &tools).await;

        match result {
            Ok(r) => {
                let _ = db::postgres::insert_token_usage(
                    &task.pg,
                    &task.agent_id,
                    &task.llm_model,
                    r.input_tokens,
                    r.output_tokens,
                )
                .await;
                info!(agent_id = %task.agent_id, "心跳任務完成: {}", r.text.chars().take(100).collect::<String>());
            }
            Err(e) => {
                error!(agent_id = %task.agent_id, "心跳執行失敗: {e:#}");
            }
        }
    }
}
