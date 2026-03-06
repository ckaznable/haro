mod config;

use std::path::Path;

use anyhow::{Context, Result};
use tracing::info;

use crate::channel::{self, Channel};
pub use config::{AgentConfig, ChannelConfig};

/// 一個 Agent 代表一組身份設定 + 頻道
pub struct Agent {
    pub id: String,
    pub prompt: String,
    pub channels: Vec<Box<dyn Channel>>,
}

/// 掃描 agents_path 下所有子目錄，每個子目錄就是一個 agent
///
/// ```text
/// agents/
/// ├── haro/
/// │   ├── config.toml   ← 頻道設定
/// │   └── PROMPT.md     ← 自訂提示詞
/// └── another_bot/
///     ├── config.toml
///     └── PROMPT.md
/// ```
pub fn load_agents(agents_path: &str) -> Result<Vec<Agent>> {
    let base = Path::new(agents_path);
    if !base.is_dir() {
        return Ok(vec![]);
    }

    let mut entries: Vec<_> = std::fs::read_dir(base)
        .with_context(|| format!("無法讀取 agents 目錄: {}", base.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut agents = Vec::new();

    for entry in entries {
        let path = entry.path();
        let id = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("無效的 agent 目錄名稱")?
            .to_owned();

        let config_path = path.join("config.toml");
        let prompt_path = path.join("PROMPT.md");

        let agent_cfg: AgentConfig = toml::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("無法讀取 {}", config_path.display()))?,
        )
        .with_context(|| format!("解析 {} 失敗", config_path.display()))?;

        let prompt = std::fs::read_to_string(&prompt_path).unwrap_or_default();
        let channels = build_channels(&agent_cfg.channels)?;

        info!(agent_id = %id, channels = channels.len(), "載入 agent");
        agents.push(Agent { id, prompt, channels });
    }

    Ok(agents)
}

fn resolve_bot_token(cfg: &ChannelConfig) -> Result<String> {
    if let Some(env_name) = &cfg.bot_token_env {
        if let Ok(v) = std::env::var(env_name) {
            let v = v.trim().to_owned();
            if !v.is_empty() {
                return Ok(v);
            }
        }
    }
    if let Some(token) = &cfg.bot_token {
        if !token.is_empty() {
            return Ok(token.clone());
        }
    }
    anyhow::bail!("需要設定 bot_token 或 bot_token_env")
}

pub fn build_channels(configs: &[ChannelConfig]) -> Result<Vec<Box<dyn Channel>>> {
    let mut channels: Vec<Box<dyn Channel>> = Vec::new();

    for cfg in configs {
        match cfg.channel_type.as_str() {
            "telegram" => {
                let token = resolve_bot_token(cfg)?;
                channels.push(Box::new(channel::telegram::TelegramChannel::new(
                    &token,
                    &cfg.allowed_users,
                )));
            }
            other => anyhow::bail!("未知的 channel 類型: {other}"),
        }
    }

    Ok(channels)
}
