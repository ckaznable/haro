mod agent;
mod api;
mod channel;
mod cli;
mod config;
mod db;
mod models;
mod runner;
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
    let brain_cfg = cfg.brain.as_ref().unwrap_or(&cfg.llm);
    let brain = Arc::new(api::gemini::GeminiProvider::new(
        &brain_cfg.api_key,
        &brain_cfg.model,
        brain_cfg.dimension,
        brain_cfg.thinking_budget,
        brain_cfg.thinking_level.as_deref(),
    ));

    let res = runner::SharedResources {
        pg: Arc::new(pg_pool),
        qdrant: Arc::new(qdrant),
        embedder: Arc::new(embedder),
        llm: Arc::new(llm),
        worker: Arc::new(worker),
        brain,
    };

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
                mode: agent::AgentMode::default(),
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

    let handles = runner::spawn_all(agents, &res, &cfg);

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
                    images: &[],
                    json_mode: false,
                    temperature: 0.7,
                })
                .await?;
            Ok(Some(result.text))
        })
    });

    let commands = Arc::new(CommandRegistry::new());
    Box::new(channel::cli::CliChannel)
        .start(handler, commands)
        .await
}
