pub mod cli;
pub mod telegram;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;

/// 收到的訊息（跨平台共用）
#[derive(Debug, Clone)]
pub struct IncomingMessage {
    /// 來源頻道的唯一識別
    pub bot_id: String,
    /// 傳送者識別
    pub sender_id: String,
    /// 訊息文本
    pub text: String,
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
pub type MessageHandler = Arc<
    dyn Fn(IncomingMessage) -> Pin<Box<dyn Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
>;

/// 訊息頻道抽象（Telegram / Discord / LINE / ...）
///
/// `start` 消耗 channel 並啟動監聽迴圈，收到訊息時透過 handler 處理
pub trait Channel: Send + Sync {
    fn start(
        self: Box<Self>,
        handler: MessageHandler,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>;
}
