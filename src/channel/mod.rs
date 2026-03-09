pub mod cli;
pub mod telegram;

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;

/// 圖片資料（跨平台共用）
#[derive(Debug, Clone)]
pub struct ImageData {
    pub mime_type: String,
    pub data: Vec<u8>,
    /// 平台專屬檔案識別（Telegram file_id）
    pub file_id: Option<String>,
    /// 來源 chat ID（Telegram chat_id）
    pub source_chat_id: Option<i64>,
    /// 來源訊息 ID（Telegram message_id）
    pub source_message_id: Option<i32>,
}

/// 收到的訊息（跨平台共用）
#[derive(Debug, Clone)]
pub struct IncomingMessage {
    /// 來源頻道的唯一識別
    pub bot_id: String,
    /// 傳送者識別
    pub sender_id: String,
    /// 訊息文本
    pub text: String,
    /// 附帶的圖片
    pub images: Vec<ImageData>,
    /// 用於回覆的 handle（平台各自實作）
    pub reply_handle: ReplyHandle,
}

/// 回覆用 handle，不同平台有不同的 variant
#[derive(Debug, Clone)]
pub enum ReplyHandle {
    Telegram { chat_id: i64 },
    Cli,
}

/// 訊息處理函式型別（object-safe）
/// 回傳 `Some(text)` 表示回覆文字，`None` 表示靜默處理（Telegram 會改用 emoji reaction）
pub type MessageHandler = Arc<
    dyn Fn(IncomingMessage) -> Pin<Box<dyn Future<Output = Result<Option<String>>> + Send>>
        + Send
        + Sync,
>;

/// 指令處理函式型別：接收 (sender_id, args)，回傳回應文字
pub type CommandFn = Arc<
    dyn Fn(String, String) -> Pin<Box<dyn Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
>;

/// 已註冊的指令定義
struct CommandEntry {
    description: String,
    usage: String,
    handler: CommandFn,
}

/// 指令註冊表，各平台依自身慣例解析指令（Telegram: /cmd, CLI: /cmd, etc.）
pub struct CommandRegistry {
    commands: HashMap<String, CommandEntry>,
    /// 保持插入順序的 key 列表（HashMap 不保序）
    order: Vec<String>,
}

impl CommandRegistry {
    pub fn new() -> Self {
        Self {
            commands: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// 註冊指令
    /// - `name`: 指令名稱（不含 `/`）
    /// - `description`: 簡短說明
    /// - `usage`: 用法範例，例如 `/ping` 或 `/heartbeat set <內容>`
    /// - `handler`: 處理函式
    pub fn register(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        usage: impl Into<String>,
        handler: CommandFn,
    ) {
        let name = name.into();
        if !self.commands.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.commands.insert(
            name,
            CommandEntry {
                description: description.into(),
                usage: usage.into(),
                handler,
            },
        );
    }

    /// 嘗試匹配指令，回傳 handler
    pub fn resolve(&self, name: &str) -> Option<&CommandFn> {
        self.commands.get(name).map(|e| &e.handler)
    }

    /// 產生 help 文字，列出所有已註冊指令
    pub fn help_text(&self) -> String {
        self.order
            .iter()
            .filter_map(|name| {
                self.commands
                    .get(name)
                    .map(|e| format!("{} — {}", e.usage, e.description))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// 訊息頻道抽象（Telegram / Discord / LINE / ...）
///
/// `start` 消耗 channel 並啟動監聽迴圈，收到訊息時透過 handler 處理
pub trait Channel: Send + Sync {
    fn start(
        self: Box<Self>,
        handler: MessageHandler,
        commands: Arc<CommandRegistry>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>;
}
