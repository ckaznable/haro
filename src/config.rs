use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::info;

use crate::agent::ChannelConfig;

/// 頂層設定，對應 config.toml
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    /// Agent 設定檔搜尋路徑
    pub agents_path: Option<String>,
    /// Fallback channels（當 agents_path 無 agent 時，enable=true 的會以預設 agent 啟動）
    #[serde(default)]
    pub channels: Vec<ChannelConfig>,
    pub database: DatabaseConfig,
    #[serde(default)]
    pub qdrant: QdrantConfig,
    pub embedding: ProviderConfig,
    pub llm: ProviderConfig,
    pub worker: ProviderConfig,
    /// 大腦模型（處理複雜任務），未設定則使用 llm
    pub brain: Option<ProviderConfig>,
    /// 心跳間隔（秒），預設 3600
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval: u64,
}

fn default_heartbeat_interval() -> u64 {
    3600
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_owned(),
        }
    }
}

/// 通用 Provider 設定（embedding / llm / worker 共用結構）
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    pub provider: String,
    #[serde(default)]
    pub api_key: String,
    /// 若有指定，從此名稱的環境變數讀取 api_key（優先於 api_key 欄位）
    pub api_key_env: Option<String>,
    pub model: String,
    /// Embedding 向量維度
    pub dimension: Option<u64>,
    /// Gemini 2.5: thinking token 上限 (0=關閉, -1=dynamic)
    pub thinking_budget: Option<i32>,
    /// Gemini 3.x: thinking 等級 ("minimal" / "low" / "medium" / "high")
    pub thinking_level: Option<String>,
}

impl ProviderConfig {
    /// 若設定了 api_key_env，從該環境變數讀取 api_key
    fn resolve_api_key_env(&mut self, section: &str) -> Result<()> {
        if let Some(env_name) = &self.api_key_env {
            match std::env::var(env_name) {
                Ok(v) => self.api_key = v,
                Err(_) if self.api_key.is_empty() => {
                    anyhow::bail!(
                        "{section}.api_key_env 指向 {env_name}，但該環境變數未設定"
                    );
                }
                Err(_) => {} // env 沒設定但 api_key 有值，保留原值
            }
        }
        Ok(())
    }
}

const DEFAULT_CONFIG: &str = include_str!("../templates/config.toml");

impl AppConfig {
    /// 產生預設 config.toml
    pub fn init() -> Result<()> {
        let path = "config.toml";
        if std::path::Path::new(path).exists() {
            anyhow::bail!("config.toml 已存在，若要重新產生請先刪除");
        }
        std::fs::write(path, DEFAULT_CONFIG).context("寫入 config.toml 失敗")?;
        info!("已產生 {path}");
        Ok(())
    }

    /// 從 config.toml 載入，敏感欄位支援環境變數覆蓋
    pub fn load() -> Result<Self> {
        let path = std::env::var("HARO_CONFIG").unwrap_or_else(|_| "config.toml".to_owned());
        let content =
            std::fs::read_to_string(&path).with_context(|| format!("無法讀取設定檔: {path}"))?;
        let mut cfg: Self =
            toml::from_str(&content).with_context(|| format!("解析設定檔失敗: {path}"))?;

        // 環境變數覆蓋
        if let Ok(v) = std::env::var("DATABASE_URL") {
            cfg.database.url = v;
        }
        if let Ok(v) = std::env::var("QDRANT_URL") {
            cfg.qdrant.url = v;
        }

        // api_key_env 解析
        cfg.embedding.resolve_api_key_env("embedding")?;
        cfg.llm.resolve_api_key_env("llm")?;
        cfg.worker.resolve_api_key_env("worker")?;
        if let Some(brain) = &mut cfg.brain {
            brain.resolve_api_key_env("brain")?;
        }

        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.embedding.api_key.is_empty() {
            anyhow::bail!(
                "embedding.api_key 未設定（config.toml 直接填寫，或透過 api_key_env 指定環境變數名稱）"
            );
        }
        if self.llm.api_key.is_empty() {
            anyhow::bail!(
                "llm.api_key 未設定（config.toml 直接填寫，或透過 api_key_env 指定環境變數名稱）"
            );
        }
        if self.worker.api_key.is_empty() {
            anyhow::bail!(
                "worker.api_key 未設定（config.toml 直接填寫，或透過 api_key_env 指定環境變數名稱）"
            );
        }
        Ok(())
    }
}
