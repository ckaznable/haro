mod agent;
mod api;
mod channel;
mod cli;
mod config;
mod db;
mod models;
mod search;
mod tool;

use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::api::LlmProvider;
use crate::channel::{Channel, CommandRegistry, MessageHandler};

#[derive(Parser)]
#[command(name = "haro", about = "Haro — LLM 記憶助手")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// 產生預設 config.toml
    Init,
    /// 建立新的 agent 目錄
    InitAgent(cli::InitAgentArgs),
    /// CLI 測試模式（直接與 LLM 互動）
    Cli,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("haro=info")),
        )
        .init();

    let args = Cli::parse();

    match args.command {
        Some(Commands::Init) => return config::AppConfig::init(),
        Some(Commands::InitAgent(init_args)) => return cli::init_agent(init_args),
        Some(Commands::Cli) => {
            let cfg = config::AppConfig::load()?;
            return run_cli(&cfg).await;
        }
        None => {}
    }

    info!("Haro 啟動中...");

    let cfg = config::AppConfig::load()?;

    // 初始化資料庫
    let pg_pool = db::postgres::init_pool(&cfg.database.url).await?;
    let vector_dim = cfg.embedding.dimension.unwrap_or(768);
    let qdrant = db::qdrant::init_client(&cfg.qdrant.url, vector_dim).await?;

    // 初始化 Provider
    let embedder = api::gemini::GeminiProvider::new(
        &cfg.embedding.api_key,
        &cfg.embedding.model,
        cfg.embedding.dimension,
        cfg.embedding.thinking_budget,
        cfg.embedding.thinking_level.as_deref(),
    );
    let llm = api::gemini::GeminiProvider::new(
        &cfg.llm.api_key,
        &cfg.llm.model,
        cfg.llm.dimension,
        cfg.llm.thinking_budget,
        cfg.llm.thinking_level.as_deref(),
    );
    let worker = api::gemini::GeminiProvider::new(
        &cfg.worker.api_key,
        &cfg.worker.model,
        cfg.worker.dimension,
        cfg.worker.thinking_budget,
        cfg.worker.thinking_level.as_deref(),
    );

    info!(
        "基礎設施就緒 — PG pool={} / Qdrant='{}' / llm={} / worker={}",
        pg_pool.size(),
        db::qdrant::COLLECTION_NAME,
        cfg.llm.model,
        cfg.worker.model,
    );

    // 初始化大腦 Provider（未設定則與 llm 共用）
    let brain = if let Some(ref brain_cfg) = cfg.brain {
        Arc::new(api::gemini::GeminiProvider::new(
            &brain_cfg.api_key,
            &brain_cfg.model,
            brain_cfg.dimension,
            brain_cfg.thinking_budget,
            brain_cfg.thinking_level.as_deref(),
        ))
    } else {
        Arc::new(api::gemini::GeminiProvider::new(
            &cfg.llm.api_key,
            &cfg.llm.model,
            cfg.llm.dimension,
            cfg.llm.thinking_budget,
            cfg.llm.thinking_level.as_deref(),
        ))
    };

    let pg_pool = Arc::new(pg_pool);
    let qdrant = Arc::new(qdrant);
    let embedder = Arc::new(embedder);
    let llm = Arc::new(llm);
    let worker = Arc::new(worker);

    // 載入所有 Agent（優先從 agents_path，fallback 到 config.toml [[channels]]）
    let mut agents = if let Some(path) = &cfg.agents_path {
        agent::load_agents(path)?
    } else {
        Vec::new()
    };

    if agents.is_empty() {
        let enabled: Vec<_> = cfg.channels.iter().filter(|c| c.enable).cloned().collect();
        if !enabled.is_empty() {
            let channels = agent::build_channels(&enabled)?;
            info!(channels = channels.len(), "使用 config.toml 預設頻道啟動");
            agents.push(agent::Agent {
                id: "default".into(),
                path: None,
                prompt: String::new(),
                soul: String::new(),
                channels,
                heartbeat: String::new(),
                brain_heartbeat: String::new(),
                llm_commands: Vec::new(),
            });
        }
    }

    if agents.is_empty() {
        anyhow::bail!(
            "沒有找到任何 agent（agents_path 未設定或為空，且 config.toml 無啟用的 [[channels]]）",
        );
    }

    info!("共載入 {} 個 agent", agents.len());

    // 啟動所有 Agent 的所有 Channel
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
            let pg = Arc::clone(&pg_pool);
            let wk = Arc::clone(&worker);
            let llm_hb = Arc::clone(&llm);
            let aid = agent_id.clone();
            let prompt = agent_prompt.clone();
            let llm_model = cfg.llm.model.clone();

            info!(agent_id = %aid, interval = heartbeat_interval, "啟動 LLM 心跳任務");
            handles.push(tokio::spawn(async move {
                run_heartbeat(
                    pg,
                    wk,
                    llm_hb,
                    aid,
                    prompt,
                    agent_heartbeat,
                    heartbeat_interval,
                    llm_model,
                )
                .await
            }));
        }

        // 啟動大腦心跳任務
        if !agent_brain_heartbeat.is_empty() {
            let pg = Arc::clone(&pg_pool);
            let wk = Arc::clone(&worker);
            let brain_hb = Arc::clone(&brain);
            let aid = agent_id.clone();
            let prompt = agent_prompt.clone();
            let brain_model = brain_model.clone();

            info!(agent_id = %aid, interval = heartbeat_interval, "啟動大腦心跳任務");
            handles.push(tokio::spawn(async move {
                run_heartbeat(
                    pg,
                    wk,
                    brain_hb,
                    aid,
                    prompt,
                    agent_brain_heartbeat,
                    heartbeat_interval,
                    brain_model,
                )
                .await
            }));
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
            let hb_path = apath.join("HEARTBEAT.md");
            cmd_registry.register(
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
            cmd_registry.register(
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

        // LLM 互動指令（來自 cmd.toml）
        for cmd_def in &agent_llm_commands {
            let cmd_prompt = cmd_def.prompt.clone();
            let cmd_tools = cmd_def.tools.clone();
            let llm_for_cmd = Arc::clone(&llm);
            let pg_for_cmd = Arc::clone(&pg_pool);
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
                            api::chat_with_tools(
                                llm.as_ref(),
                                &system,
                                &user_message,
                                &tools,
                            )
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
            Arc::new(|_sender, _args| Box::pin(async { Ok(String::new()) })), // placeholder
        );
        let help_text = cmd_registry.help_text();
        // 用實際 handler 覆蓋 placeholder
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
            let pg = Arc::clone(&pg_pool);
            let qd = Arc::clone(&qdrant);
            let emb = Arc::clone(&embedder);
            let llm_ref = Arc::clone(&llm);
            let wk = Arc::clone(&worker);
            let aid = agent_id.clone();
            let prompt = agent_prompt.clone();
            let soul = agent_soul.clone();
            let wk_model = cfg.worker.model.clone();
            let llm_model = cfg.llm.model.clone();

            let handler: MessageHandler = Arc::new(move |msg| {
                let pg = Arc::clone(&pg);
                let qd = Arc::clone(&qd);
                let emb = Arc::clone(&emb);
                let llm_ref = Arc::clone(&llm_ref);
                let wk = Arc::clone(&wk);
                let aid = aid.clone();
                let prompt = prompt.clone();
                let soul = soul.clone();
                let wk_model = wk_model.clone();
                let llm_model = llm_model.clone();

                Box::pin(async move {
                    // 0. 載入備忘錄
                    let memo = db::postgres::get_scratchpad(&pg, &aid)
                        .await?
                        .unwrap_or_default();

                    // 1. 入庫（worker 蒸餾 + embedder 向量化）
                    let (_, wk_usage) =
                        search::ingest(&pg, &qd, emb.as_ref(), wk.as_ref(), &aid, &msg.text)
                            .await?;

                    // 記錄 worker token 用量
                    db::postgres::insert_token_usage(
                        &pg,
                        &aid,
                        &wk_model,
                        wk_usage.input_tokens,
                        wk_usage.output_tokens,
                    )
                    .await?;

                    // 2. 混合搜尋取回相關記憶
                    let results =
                        search::retrieve(&pg, &qd, emb.as_ref(), &aid, &msg.text, 5).await?;

                    // 3. 載入近期歷史（填充至 ~5000 token）
                    let recent = db::postgres::get_recent_messages(&pg, &aid, 50).await?;
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
                    history_parts.reverse(); // 倒序 → 時間正序
                    let history = if history_parts.is_empty() {
                        "（無近期歷史）".to_owned()
                    } else {
                        history_parts.join("\n---\n")
                    };

                    // 4. 建立工具
                    let mut tools = tool::ToolRegistry::new();
                    for t in tool::scratchpad::tools(Arc::clone(&pg), aid.clone()) {
                        tools.register(t);
                    }
                    tools.register(tool::fetch::tool());

                    // 5. 組合上下文，用主 LLM 帶工具生成回答
                    let context: String = results
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

                    let soul_section = if soul.is_empty() {
                        String::new()
                    } else {
                        format!("## 性格設定\n{soul}\n\n")
                    };

                    let system = format!(
                        "你是一個具有長期記憶的 AI 助手。\n\n\
                         目前時間：{now}\n\n\
                         {soul_section}\
                         {prompt}\n\n\
                         ## 備忘錄 (Scratchpad)\n{memo_section}\n\n\
                         ## 相關記憶\n{context}\n\n\
                         ## 近期對話歷史\n{history}\n\n\
                         ## 指示\n\
                         請根據備忘錄、歷史記憶和近期對話與使用者的新訊息，給出自然、有幫助的回答。\n\
                         如果記憶中沒有相關資訊，就正常回答即可。\n\
                         如果有重要的事情需要記住（如使用者偏好、關鍵決定、待辦事項），\
                         請使用 save_memo 工具儲存到備忘錄中。"
                    );

                    let result =
                        api::chat_with_tools(llm_ref.as_ref(), &system, &msg.text, &tools).await?;

                    // 記錄 LLM token 用量
                    db::postgres::insert_token_usage(
                        &pg,
                        &aid,
                        &llm_model,
                        result.input_tokens,
                        result.output_tokens,
                    )
                    .await?;

                    Ok(result.text)
                })
            });

            let cmds = Arc::clone(&commands);
            handles.push(tokio::spawn(channel.start(handler, cmds)));
        }
    }

    // 等待所有頻道（正常情況下不會結束）
    for h in handles {
        match h.await {
            Err(e) => error!("頻道任務 panic: {e:#}"),
            Ok(Err(e)) => error!("頻道任務錯誤: {e:#}"),
            Ok(Ok(())) => {}
        }
    }

    Ok(())
}

const HEARTBEAT_TRIAGE_PROMPT: &str = "\
You are a triage agent. Given the heartbeat instructions, current time, and scratchpad context, \
decide whether there is a task that needs to be executed RIGHT NOW.\n\
Consider time-sensitive tasks, scheduled actions, and pending items.\n\n\
Reply with ONLY \"yes\" (needs action now) or \"no\" (skip this cycle).";

/// 心跳任務：定期喚醒 worker 判斷是否需要呼叫 LLM 執行任務
async fn run_heartbeat(
    pg: Arc<sqlx::PgPool>,
    worker: Arc<api::gemini::GeminiProvider>,
    llm: Arc<api::gemini::GeminiProvider>,
    agent_id: String,
    agent_prompt: String,
    heartbeat_prompt: String,
    interval_secs: u64,
    llm_model: String,
) -> Result<()> {
    let interval = std::time::Duration::from_secs(interval_secs);

    loop {
        tokio::time::sleep(interval).await;

        info!(agent_id = %agent_id, "心跳觸發");

        // 載入備忘錄
        let memo = db::postgres::get_scratchpad(&pg, &agent_id)
            .await
            .unwrap_or(None)
            .unwrap_or_default();

        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %:z");

        let context = format!(
            "目前時間：{now}\n\n\
             ## 心跳指示\n{heartbeat_prompt}\n\n\
             ## 備忘錄\n{memo}"
        );

        // Worker（小模型）判斷是否需要喚醒 LLM
        let triage_result = worker
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
                error!(agent_id = %agent_id, "心跳分流判斷失敗: {e:#}");
                continue;
            }
        };

        if !needs_action {
            info!(agent_id = %agent_id, "心跳：無需執行任務");
            continue;
        }

        info!(agent_id = %agent_id, "心跳：喚醒 LLM 執行任務");

        // 建立工具
        let mut tools = tool::ToolRegistry::new();
        for t in tool::scratchpad::tools(Arc::clone(&pg), agent_id.clone()) {
            tools.register(t);
        }
        tools.register(tool::fetch::tool());

        let system = format!(
            "你是一個具有長期記憶的 AI 助手。\n\n\
             目前時間：{now}\n\n\
             {agent_prompt}\n\n\
             ## 備忘錄 (Scratchpad)\n{memo}\n\n\
             ## 指示\n\
             這是一個定期心跳喚醒。請根據以下心跳指示執行任務。\n\
             如果有重要的事情需要記住，請使用 save_memo 工具儲存到備忘錄中。\n\n\
             ## 心跳指示\n{heartbeat_prompt}"
        );

        let result = api::chat_with_tools(llm.as_ref(), &system, "心跳觸發，請執行任務。", &tools).await;

        match result {
            Ok(r) => {
                let _ = db::postgres::insert_token_usage(
                    &pg,
                    &agent_id,
                    &llm_model,
                    r.input_tokens,
                    r.output_tokens,
                )
                .await;
                info!(agent_id = %agent_id, "心跳任務完成: {}", r.text.chars().take(100).collect::<String>());
            }
            Err(e) => {
                error!(agent_id = %agent_id, "心跳執行失敗: {e:#}");
            }
        }
    }
}

/// CLI 測試模式：不啟動 DB / Agent，直接與 LLM 互動
async fn run_cli(cfg: &config::AppConfig) -> Result<()> {
    let llm = Arc::new(api::gemini::GeminiProvider::new(
        &cfg.llm.api_key,
        &cfg.llm.model,
        cfg.llm.dimension,
        cfg.llm.thinking_budget,
        cfg.llm.thinking_level.as_deref(),
    ));

    info!("CLI 測試模式 — model={}", cfg.llm.model);
    eprintln!("輸入 /quit 或 /exit 離開\n");

    let handler: MessageHandler = Arc::new(move |msg| {
        let llm = Arc::clone(&llm);
        Box::pin(async move {
            let result = llm
                .generate(api::GenerateParams {
                    system: None,
                    user_message: &msg.text,
                    json_mode: false,
                    temperature: 0.7,
                })
                .await?;
            Ok(result.text)
        })
    });

    let commands = Arc::new(CommandRegistry::new());
    Box::new(channel::cli::CliChannel)
        .start(handler, commands)
        .await
}
