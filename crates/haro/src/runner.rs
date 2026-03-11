use std::sync::Arc;

use anyhow::Result;
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::agent::{Agent, AgentMode};
use crate::api::{self, LlmProvider};
use crate::channel::{CommandRegistry, MessageHandler, Notifier};
use crate::config::AppConfig;
use crate::db::postgres::ImageMeta;
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
        let agent_mode = agent.mode;
        let agent_prompt = agent.prompt;
        let agent_soul = agent.soul;
        let agent_heartbeat = agent.heartbeat;
        let agent_brain_heartbeat = agent.brain_heartbeat;
        let agent_path = agent.path;
        let agent_llm_commands = agent.llm_commands;

        // 收集所有頻道的 notifier（在 channel.start() 消耗 channel 前）
        let notifiers: Vec<Arc<dyn Notifier>> = agent
            .channels
            .iter()
            .filter_map(|ch| ch.notifier())
            .map(|n| Arc::from(n))
            .collect();

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
                notifiers: notifiers.clone(),
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
                notifiers: notifiers.clone(),
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

        // Skills 目錄（存在才啟用）
        let skills_path = agent_path
            .as_ref()
            .map(|p| p.join("skills"))
            .filter(|p| p.is_dir());

        // /skills 指令：列出可用 skills
        if let Some(ref sp) = skills_path {
            let sp_clone = sp.clone();
            cmd_registry.register(
                "skills",
                "列出可用的 skills",
                "/skills",
                Arc::new(move |_sender, _args| {
                    let sp = sp_clone.clone();
                    Box::pin(async move {
                        let entries = tool::skills::list_skill_entries(&sp);
                        if entries.is_empty() {
                            return Ok("目前沒有可用的 skills。".into());
                        }
                        let list: Vec<String> = entries
                            .iter()
                            .map(|(name, desc)| {
                                if desc.is_empty() {
                                    format!("• {name}")
                                } else {
                                    format!("• {name} — {desc}")
                                }
                            })
                            .collect();
                        Ok(format!("可用 Skills:\n{}", list.join("\n")))
                    })
                }),
            );
        }

        // ingest 模式：註冊 /ask 指令，用於查詢向量資料庫
        if agent_mode == AgentMode::Ingest {
            let ask_ctx = Arc::new(MessageContext {
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
                image_embed: cfg.embedding.image_embed,
                skills_path: skills_path.clone(),
            });
            cmd_registry.register(
                "ask",
                "查詢已收集的資料",
                "/ask <問題>",
                Arc::new(move |_sender, args| {
                    let ctx = Arc::clone(&ask_ctx);
                    Box::pin(async move {
                        let question = args.trim();
                        if question.is_empty() {
                            return Ok("用法: /ask <問題>".into());
                        }
                        handle_query(&ctx, question, &[]).await
                    })
                }),
            );
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
                                images: &[],
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

        // ingest 模式：建立批次入庫佇列 + 背景任務
        let ingest_queue = if agent_mode == AgentMode::Ingest {
            let (queue, batch_handle) = search::spawn_batch_ingest(search::BatchIngestConfig {
                pg: Arc::clone(&res.pg),
                qdrant: Arc::clone(&res.qdrant),
                embedder: Arc::clone(&res.embedder),
                worker: Arc::clone(&res.worker),
                batch_size: 10,
                flush_interval: std::time::Duration::from_secs(5),
            });
            handles.push(batch_handle);
            info!(agent_id = %agent_id, "批次入庫背景任務已啟動");
            Some(queue)
        } else {
            None
        };

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
                image_embed: cfg.embedding.image_embed,
                skills_path: skills_path.clone(),
            });

            let mode = agent_mode.clone();
            let queue = ingest_queue.clone();
            let handler: MessageHandler = Arc::new(move |msg| {
                let ctx = Arc::clone(&msg_ctx);
                let mode = mode.clone();
                let queue = queue.clone();
                Box::pin(async move {
                    match mode {
                        AgentMode::Chat => handle_message(&ctx, &msg).await,
                        AgentMode::Ingest => {
                            if let Some(q) = &queue {
                                q.push(&ctx.agent_id, &msg.text, &ctx.worker_model).await?;
                            } else {
                                let images = to_image_inputs(&msg.images);
                                let (file_ids, source_chat_id, source_message_id) = extract_image_meta(&msg.images);
                                let image_meta = ImageMeta { file_ids: &file_ids, source_chat_id, source_message_id };
                                handle_ingest(&ctx, &msg.text, &images, &image_meta).await?;
                            }
                            Ok(None)
                        }
                    }
                })
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
    image_embed: bool,
    /// Agent 的 skills 目錄路徑（None = 無 skills）
    skills_path: Option<std::path::PathBuf>,
}

/// 將 channel::ImageData 轉換為 api::ImageInput
fn to_image_inputs(images: &[crate::channel::ImageData]) -> Vec<api::ImageInput> {
    images
        .iter()
        .map(|img| api::ImageInput {
            mime_type: img.mime_type.clone(),
            data: img.data.clone(),
        })
        .collect()
}

/// 從 channel::ImageData 提取圖片來源 metadata
fn extract_image_meta(images: &[crate::channel::ImageData]) -> (Vec<String>, Option<i64>, Option<i32>) {
    let file_ids: Vec<String> = images.iter().filter_map(|img| img.file_id.clone()).collect();
    // 取第一張圖片的來源資訊（同一則訊息的圖片來自同一個 chat/message）
    let source_chat_id = images.first().and_then(|img| img.source_chat_id);
    let source_message_id = images.first().and_then(|img| img.source_message_id);
    (file_ids, source_chat_id, source_message_id)
}

/// 僅入庫（ingest 模式），不呼叫 LLM 回覆
async fn handle_ingest(ctx: &MessageContext, text: &str, images: &[api::ImageInput], image_meta: &ImageMeta<'_>) -> Result<()> {
    let embed_images = if ctx.image_embed { images } else { &[] };
    let (_, wk_usage) =
        search::ingest(&ctx.pg, &ctx.qdrant, ctx.embedder.as_ref(), ctx.worker.as_ref(), &ctx.agent_id, text, embed_images, image_meta).await?;

    db::postgres::insert_token_usage(&ctx.pg, &ctx.agent_id, &ctx.worker_model, wk_usage.input_tokens, wk_usage.output_tokens).await?;

    Ok(())
}

/// 查詢（不入庫）：檢索記憶 + LLM 回答
async fn handle_query(ctx: &MessageContext, question: &str, images: &[api::ImageInput]) -> Result<String> {
    let memo = db::postgres::get_scratchpad(&ctx.pg, &ctx.agent_id)
        .await?
        .unwrap_or_default();

    let results = search::retrieve(&ctx.pg, &ctx.qdrant, ctx.embedder.as_ref(), &ctx.agent_id, question, 5).await?;

    let mut tools = tool::ToolRegistry::new();
    for t in tool::scratchpad::tools(Arc::clone(&ctx.pg), ctx.agent_id.clone()) {
        tools.register(t);
    }
    tools.register(tool::fetch::tool());

    // 註冊 skills 工具（如有 skills 目錄）
    let has_skills = ctx.skills_path.is_some();
    if let Some(ref sp) = ctx.skills_path {
        for t in tool::skills::tools(sp.clone()) {
            tools.register(t);
        }
    }

    let context_str: String = results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("[記憶 {}] {}", i + 1, r.original_text))
        .collect::<Vec<_>>()
        .join("\n\n");

    // 尚未蒸餾的 pending 訊息也加入上下文
    let pending_texts = db::postgres::get_pending_texts(&ctx.pg, &ctx.agent_id)
        .await
        .unwrap_or_default();
    let pending_section = if pending_texts.is_empty() {
        String::new()
    } else {
        let items: String = pending_texts
            .iter()
            .enumerate()
            .map(|(i, t)| format!("[待處理 {}] {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n\n");
        format!("\n\n## 待處理資料（尚未索引）\n{items}")
    };

    let memo_section = if memo.is_empty() { "（空）".to_owned() } else { memo };
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %:z");
    let soul_section = if ctx.soul.is_empty() {
        String::new()
    } else {
        format!("## 性格設定\n{}\n\n", ctx.soul)
    };

    let skills_instruction = if has_skills {
        "\n如果使用者要求使用某個 skill，請先使用 list_skills 工具查看可用的 skills，\
         再使用 get_skill 工具載入該 skill 的內容並依照其指示執行。"
    } else {
        ""
    };

    let system = format!(
        "你是一個知識庫助手，根據已收集的資料回答問題。\n\n\
         目前時間：{now}\n\n\
         {soul_section}\
         {}\n\n\
         ## 備忘錄 (Scratchpad)\n{memo_section}\n\n\
         ## 相關資料\n{context_str}{pending_section}\n\n\
         ## 指示\n\
         請根據上述資料回答使用者的問題。\n\
         如果資料中沒有相關內容，請如實告知。\n\
         如果有重要的事情需要記住，請使用 save_memo 工具儲存。{skills_instruction}",
        ctx.prompt
    );

    let result = api::chat_with_images(ctx.llm.as_ref(), &system, question, images, &tools).await?;

    db::postgres::insert_token_usage(&ctx.pg, &ctx.agent_id, &ctx.llm_model, result.input_tokens, result.output_tokens).await?;

    Ok(result.text)
}

/// 處理單則訊息（chat 模式：入庫 + 回覆）
async fn handle_message(
    ctx: &MessageContext,
    msg: &crate::channel::IncomingMessage,
) -> Result<Option<String>> {
    let images = to_image_inputs(&msg.images);
    let (file_ids, source_chat_id, source_message_id) = extract_image_meta(&msg.images);
    let image_meta = ImageMeta { file_ids: &file_ids, source_chat_id, source_message_id };

    // 入庫（image_embed 由 handle_ingest 內部判斷）
    handle_ingest(ctx, &msg.text, &images, &image_meta).await?;

    // 查詢 + 回覆（LLM chat 端一律傳圖，不受 image_embed 影響）
    handle_query(ctx, &msg.text, &images).await.map(Some)
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
    notifiers: Vec<Arc<dyn Notifier>>,
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
                images: &[],
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
        if !task.notifiers.is_empty() {
            tools.register(tool::notify::tool(task.notifiers.clone()));
        }

        let notify_instruction = if task.notifiers.is_empty() {
            ""
        } else {
            "\n如果需要主動通知使用者，請使用 send_message 工具發送訊息到頻道。"
        };

        let system = format!(
            "你是一個具有長期記憶的 AI 助手。\n\n\
             目前時間：{now}\n\n\
             {}\n\n\
             ## 備忘錄 (Scratchpad)\n{memo}\n\n\
             ## 指示\n\
             這是一個定期心跳喚醒。請根據以下心跳指示執行任務。\n\
             如果有重要的事情需要記住，請使用 save_memo 工具儲存到備忘錄中。{notify_instruction}\n\n\
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
