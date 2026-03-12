pub mod cli;
pub mod telegram;

use std::collections::HashMap;
use std::future::Future;
use std::io::Cursor;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use image::ImageReader;

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

/// 頻道通知器：可主動向頻道發送訊息（用於心跳任務等）
pub trait Notifier: Send + Sync {
    /// 發送訊息到頻道
    fn send<'a>(&'a self, message: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

/// 訊息頻道抽象（Telegram / Discord / LINE / ...）
///
/// `start` 消耗 channel 並啟動監聽迴圈，收到訊息時透過 handler 處理
pub trait Channel: Send + Sync {
    /// 建立通知器（在 start 消耗 channel 前呼叫），回傳 None 表示此頻道不支援主動通知
    fn notifier(&self) -> Option<Box<dyn Notifier>>;

    fn start(
        self: Box<Self>,
        handler: MessageHandler,
        commands: Arc<CommandRegistry>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>;
}

/// 允許直接傳遞的 MIME type（不需格式轉換）
const ALLOWED_MIME_TYPES: &[&str] = &[
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
];

/// 最大長邊像素
const MAX_DIMENSION: u32 = 1536;

/// 處理圖片：超過 1536px 則等比縮放，非允許格式則轉為 JPEG。
/// 回傳 (mime_type, data)。無法解碼時原樣回傳。
pub fn process_image(mime_type: &str, data: Vec<u8>) -> (String, Vec<u8>) {
    // HEIC/HEIF 無法由 image crate 解碼，直接放行
    if mime_type == "image/heic" || mime_type == "image/heif" {
        return (mime_type.to_owned(), data);
    }

    let reader = match ImageReader::new(Cursor::new(&data)).with_guessed_format() {
        Ok(r) => r,
        Err(_) => return (mime_type.to_owned(), data),
    };

    let mut img = match reader.decode() {
        Ok(i) => i,
        Err(_) => return (mime_type.to_owned(), data),
    };

    // 等比縮放
    let (w, h) = (img.width(), img.height());
    let max_side = w.max(h);
    if max_side > MAX_DIMENSION {
        let new_w = (w as f64 * MAX_DIMENSION as f64 / max_side as f64).round() as u32;
        let new_h = (h as f64 * MAX_DIMENSION as f64 / max_side as f64).round() as u32;
        img = img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3);
        tracing::info!(original = %format!("{w}x{h}"), resized = %format!("{new_w}x{new_h}"), "圖片縮放");
    }

    // 決定輸出格式
    let need_convert = !ALLOWED_MIME_TYPES.contains(&mime_type);
    let resized = max_side > MAX_DIMENSION;

    if !need_convert && !resized {
        return (mime_type.to_owned(), data);
    }

    let (out_format, out_mime) = if need_convert {
        (image::ImageFormat::Jpeg, "image/jpeg")
    } else if mime_type == "image/png" {
        (image::ImageFormat::Png, "image/png")
    } else if mime_type == "image/webp" {
        (image::ImageFormat::WebP, "image/webp")
    } else {
        (image::ImageFormat::Jpeg, "image/jpeg")
    };

    let mut buf = Cursor::new(Vec::new());
    match img.write_to(&mut buf, out_format) {
        Ok(()) => (out_mime.to_owned(), buf.into_inner()),
        Err(e) => {
            tracing::warn!("圖片編碼失敗，使用原始資料: {e}");
            (mime_type.to_owned(), data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_handler() -> CommandFn {
        Arc::new(|_sender, _args| Box::pin(async { Ok("ok".into()) }))
    }

    #[test]
    fn register_and_resolve() {
        let mut reg = CommandRegistry::new();
        reg.register("ping", "check", "/ping", noop_handler());

        assert!(reg.resolve("ping").is_some());
        assert!(reg.resolve("unknown").is_none());
    }

    #[test]
    fn help_text_contains_all_commands() {
        let mut reg = CommandRegistry::new();
        reg.register("ping", "檢查在線", "/ping", noop_handler());
        reg.register("help", "顯示幫助", "/help", noop_handler());

        let help = reg.help_text();
        assert!(help.contains("/ping — 檢查在線"));
        assert!(help.contains("/help — 顯示幫助"));
    }

    #[test]
    fn help_text_preserves_insertion_order() {
        let mut reg = CommandRegistry::new();
        reg.register("aaa", "first", "/aaa", noop_handler());
        reg.register("zzz", "second", "/zzz", noop_handler());
        reg.register("mmm", "third", "/mmm", noop_handler());

        let help = reg.help_text();
        let lines: Vec<&str> = help.lines().collect();
        assert!(lines[0].contains("aaa"));
        assert!(lines[1].contains("zzz"));
        assert!(lines[2].contains("mmm"));
    }

    #[test]
    fn register_override_replaces_handler() {
        let mut reg = CommandRegistry::new();
        reg.register("test", "v1", "/test", noop_handler());
        reg.register("test", "v2", "/test v2", noop_handler());

        let help = reg.help_text();
        // 覆蓋後只出現一次，使用新的 description
        assert!(help.contains("v2"));
        assert_eq!(help.matches("test").count(), 1);
    }

    #[tokio::test]
    async fn command_handler_receives_args() {
        let mut reg = CommandRegistry::new();
        reg.register(
            "echo",
            "echo",
            "/echo",
            Arc::new(|_sender, args| {
                Box::pin(async move { Ok(format!("got: {args}")) })
            }),
        );

        let handler = reg.resolve("echo").unwrap();
        let result = handler("user".into(), "hello world".into()).await.unwrap();
        assert_eq!(result, "got: hello world");
    }
}
