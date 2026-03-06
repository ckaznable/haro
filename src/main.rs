mod agent;
mod api;
mod channel;
mod config;
mod db;
mod models;
mod search;
mod tool;

use std::sync::Arc;

use anyhow::Result;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::api::LlmProvider;
use crate::channel::{Channel, MessageHandler};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("haro=info")),
        )
        .init();

    info!("Haro 啟動中...");

    if std::env::args().any(|a| a == "--init") {
        return config::AppConfig::init();
    }

    let cli_mode = std::env::args().any(|a| a == "--cli");

    let cfg = config::AppConfig::load()?;

    if cli_mode {
        return run_cli(&cfg).await;
    }

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
                prompt: String::new(),
                channels,
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

    for agent in agents {
        let agent_id = agent.id;
        let agent_prompt = agent.prompt;

        for channel in agent.channels {
            let pg = Arc::clone(&pg_pool);
            let qd = Arc::clone(&qdrant);
            let emb = Arc::clone(&embedder);
            let llm_ref = Arc::clone(&llm);
            let wk = Arc::clone(&worker);
            let aid = agent_id.clone();
            let prompt = agent_prompt.clone();
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

                    let system = format!(
                        "你是一個具有長期記憶的 AI 助手。\n\n\
                         目前時間：{now}\n\n\
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

            handles.push(tokio::spawn(channel.start(handler)));
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

    Box::new(channel::cli::CliChannel).start(handler).await
}
