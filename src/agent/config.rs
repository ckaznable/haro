use serde::Deserialize;

/// Agent 層級設定（agents/<id>/config.toml）
#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    #[serde(default)]
    pub channels: Vec<ChannelConfig>,
}

/// 單一頻道設定
#[derive(Debug, Clone, Deserialize)]
pub struct ChannelConfig {
    /// 是否啟用（主要用於 config.toml 的 fallback channels）
    #[serde(default)]
    pub enable: bool,
    #[serde(alias = "type", alias = "channel_type")]
    pub channel_type: String,
    pub bot_token: Option<String>,
    /// 從此環境變數讀取 bot_token（優先於 bot_token）
    pub bot_token_env: Option<String>,
    /// 允許的使用者（user ID 或 @username），空陣列 = 不限制
    #[serde(default)]
    pub allowed_users: Vec<String>,
}
