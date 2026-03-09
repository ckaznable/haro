use serde::Deserialize;

/// Agent 層級設定（agents/<id>/config.toml）
#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    #[serde(default)]
    pub channels: Vec<ChannelConfig>,
}

/// LLM 互動指令設定（agents/<id>/cmd.toml）
#[derive(Debug, Clone, Deserialize)]
pub struct CmdConfig {
    #[serde(default)]
    pub commands: Vec<LlmCmdDef>,
}

/// 單一 LLM 指令定義
#[derive(Debug, Clone, Deserialize)]
pub struct LlmCmdDef {
    /// 指令名稱（不含 `/`）
    pub name: String,
    /// 指令說明
    pub description: String,
    /// 注入給 LLM 的提示詞
    pub prompt: String,
    /// 可使用的工具名稱列表（如 ["fetch", "save_memo"]），空 = 不使用工具
    #[serde(default)]
    pub tools: Vec<String>,
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
