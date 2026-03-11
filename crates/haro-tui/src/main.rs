mod db;
mod ui;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(name = "haro-tui", about = "Haro TUI — 資料庫檢視器")]
struct Cli {
    /// 資料庫連線字串（預設從 config.toml 讀取）
    #[arg(long)]
    database_url: Option<String>,
}

/// 從 haro config.toml 讀取 database.url
fn load_database_url() -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Cfg {
        database: Db,
    }
    #[derive(serde::Deserialize)]
    struct Db {
        url: String,
    }

    // 嘗試 HARO_CONFIG → config.toml
    let path = std::env::var("HARO_CONFIG").unwrap_or_else(|_| "config.toml".into());
    let content = std::fs::read_to_string(&path).ok()?;
    let cfg: Cfg = toml::from_str(&content).ok()?;
    Some(cfg.database.url)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    let database_url = args
        .database_url
        .or_else(|| std::env::var("DATABASE_URL").ok())
        .or_else(load_database_url)
        .ok_or_else(|| anyhow::anyhow!(
            "需要資料庫連線：--database-url 或 DATABASE_URL 環境變數或 config.toml"
        ))?;

    let pool = db::connect(&database_url).await?;

    ui::run(pool).await
}
