mod config;

use std::path::Path;

use anyhow::{Context, Result};
use tracing::info;

use crate::channel::{self, Channel};
pub use config::{AgentConfig, ChannelConfig, LlmCmdDef};

/// 一個 Agent 代表一組身份設定 + 頻道
pub struct Agent {
    pub id: String,
    /// Agent 目錄路徑（用於讀寫 HEARTBEAT.md 等檔案）
    pub path: Option<std::path::PathBuf>,
    pub prompt: String,
    /// 性格設定（來自 SOUL.md），空字串表示不啟用
    pub soul: String,
    pub channels: Vec<Box<dyn Channel>>,
    /// 心跳提示詞（來自 HEARTBEAT.md），空字串表示不啟用心跳
    pub heartbeat: String,
    /// 大腦心跳提示詞（來自 BRAIN_HEARTBEAT.md），空字串表示不啟用
    pub brain_heartbeat: String,
    /// LLM 互動指令（來自 cmd.toml）
    pub llm_commands: Vec<LlmCmdDef>,
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
        let soul = std::fs::read_to_string(path.join("SOUL.md"))
            .unwrap_or_default()
            .trim()
            .to_owned();
        let heartbeat = std::fs::read_to_string(path.join("HEARTBEAT.md"))
            .unwrap_or_default()
            .trim()
            .to_owned();
        let brain_heartbeat = std::fs::read_to_string(path.join("BRAIN_HEARTBEAT.md"))
            .unwrap_or_default()
            .trim()
            .to_owned();
        let channels = build_channels(&agent_cfg.channels)?;

        let cmd_path = path.join("cmd.toml");
        let llm_commands = if cmd_path.exists() {
            let cmd_cfg: config::CmdConfig = toml::from_str(
                &std::fs::read_to_string(&cmd_path)
                    .with_context(|| format!("無法讀取 {}", cmd_path.display()))?,
            )
            .with_context(|| format!("解析 {} 失敗", cmd_path.display()))?;
            cmd_cfg.commands
        } else {
            Vec::new()
        };

        info!(
            agent_id = %id,
            channels = channels.len(),
            heartbeat = !heartbeat.is_empty(),
            brain_heartbeat = !brain_heartbeat.is_empty(),
            llm_commands = llm_commands.len(),
            "載入 agent"
        );
        agents.push(Agent {
            id,
            path: Some(path),
            prompt,
            soul,
            channels,
            heartbeat,
            brain_heartbeat,
            llm_commands,
        });
    }

    Ok(agents)
}

fn resolve_bot_token(cfg: &ChannelConfig) -> Result<String> {
    if let Some(env_name) = &cfg.bot_token_env
        && let Ok(v) = std::env::var(env_name)
    {
        let v = v.trim().to_owned();
        if !v.is_empty() {
            return Ok(v);
        }
    }
    if let Some(token) = &cfg.bot_token
        && !token.is_empty()
    {
        return Ok(token.clone());
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
