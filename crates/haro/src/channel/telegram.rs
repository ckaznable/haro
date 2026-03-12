use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use teloxide::net::Download;
use teloxide::prelude::*;
use teloxide::types::{BotCommand, ChatAction, ParseMode, ReactionType};
use tokio::task::AbortHandle;
use tracing::{error, info, warn};

use super::{Channel, CommandRegistry, ImageData, IncomingMessage, MessageHandler, Notifier, ReplyHandle};

/// 活躍任務的 abort handles（handler + typing）
type ActiveTask = Arc<std::sync::Mutex<Option<(AbortHandle, AbortHandle)>>>;

/// 動態收集的已知 chat ID（從收到的訊息中學習）
type KnownChatIds = Arc<std::sync::Mutex<HashSet<i64>>>;

/// Telegram Bot 頻道實作
pub struct TelegramChannel {
    bot: Bot,
    allowed_users: Arc<HashSet<String>>,
    /// 動態收集的 chat_id（配合 @username 使用，訊息進來時自動記錄）
    known_chat_ids: KnownChatIds,
}

impl TelegramChannel {
    /// `allowed_users` 可放 user ID（數字）或 @username，空 = 不限制
    pub fn new(bot_token: &str, allowed_users: &[String]) -> Self {
        // 預先填入 allowed_users 中的純數字 ID
        let pre_known: HashSet<i64> = allowed_users
            .iter()
            .filter_map(|u| u.parse::<i64>().ok())
            .collect();
        Self {
            bot: Bot::new(bot_token),
            allowed_users: Arc::new(allowed_users.iter().cloned().collect()),
            known_chat_ids: Arc::new(std::sync::Mutex::new(pre_known)),
        }
    }
}

/// 嘗試以 MarkdownV2 發送，若失敗則 fallback 為純文字
async fn send_reply(bot: &Bot, chat_id: ChatId, text: &str) {
    if let Ok(md) = telegram_markdown_v2::convert(text) {
        let result = bot
            .send_message(chat_id, &md)
            .parse_mode(ParseMode::MarkdownV2)
            .await;

        if result.is_ok() {
            return;
        }
        warn!("MarkdownV2 發送失敗，fallback 純文字");
    }

    if let Err(e) = bot.send_message(chat_id, text).await {
        error!("Telegram 發送回覆失敗: {e}");
    }
}

/// 從 Telegram 訊息文字解析 slash command，回傳 (cmd_name, args)
/// 例: "/heartbeat show" → Some(("heartbeat", "show"))
/// 例: "/status" → Some(("status", ""))
fn parse_command(text: &str) -> Option<(&str, &str)> {
    let text = text.trim();
    if !text.starts_with('/') {
        return None;
    }
    let without_slash = &text[1..];
    // 去掉 @bot_username 後綴（如 /cmd@botname args）
    let (cmd_part, args) = without_slash
        .split_once(char::is_whitespace)
        .unwrap_or((without_slash, ""));
    let cmd = cmd_part.split('@').next().unwrap_or(cmd_part);
    Some((cmd, args.trim()))
}

/// 從 Telegram Message 取得文字（text 或 caption），轉發訊息加上來源標記，
/// 回覆訊息時帶入被回覆的文字作為引用上下文
fn extract_text(msg: &Message) -> Option<String> {
    let body = msg.text().or_else(|| msg.caption());

    let mut parts = Vec::new();

    // 轉發來源標記
    let fwd = format_forward_origin(msg);
    if !fwd.is_empty() {
        parts.push(fwd);
    }

    // 被回覆訊息的文字（引用上下文）
    if let Some(reply) = msg.reply_to_message() {
        let reply_text = reply.text().or_else(|| reply.caption());
        if let Some(rt) = reply_text {
            let quoted: String = if rt.len() > 500 {
                format!("{}…", &rt[..rt.floor_char_boundary(500)])
            } else {
                rt.to_owned()
            };
            parts.push(format!("[引用訊息] {quoted}"));
        }
    }

    if let Some(b) = body {
        parts.push(b.to_owned());
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

/// 從轉發來源產生前綴標記（非轉發訊息回傳空字串）
fn format_forward_origin(msg: &Message) -> String {
    use teloxide::types::MessageOrigin;

    let Some(origin) = msg.forward_origin() else {
        return String::new();
    };

    match origin {
        MessageOrigin::User { date, sender_user, .. } => {
            let name = sender_user.username.as_ref()
                .map(|u| format!("@{u}"))
                .unwrap_or_else(|| sender_user.full_name());
            format!("[轉發自 {name}，{date}]")
        }
        MessageOrigin::HiddenUser { date, sender_user_name, .. } => {
            format!("[轉發自 {sender_user_name}，{date}]")
        }
        MessageOrigin::Chat { date, sender_chat, .. } => {
            let name = sender_chat.username()
                .map(|u| format!("@{u}"))
                .unwrap_or_else(|| sender_chat.title().unwrap_or("未知群組").to_owned());
            format!("[轉發自 {name}，{date}]")
        }
        MessageOrigin::Channel { date, chat, .. } => {
            let name = chat.username()
                .map(|u| format!("@{u}"))
                .unwrap_or_else(|| chat.title().unwrap_or("未知頻道").to_owned());
            format!("[轉發自頻道 {name}，{date}]")
        }
    }
}

/// 下載 Telegram 圖片（選最大尺寸），回傳 ImageData
async fn download_photo(bot: &Bot, msg: &Message) -> Option<ImageData> {
    let photos = msg.photo()?;
    // Telegram 回傳多種尺寸，選最大的（最後一個）
    let largest = photos.last()?;

    let file = match bot.get_file(largest.file.id.clone()).await {
        Ok(f) => f,
        Err(e) => {
            warn!("取得圖片檔案資訊失敗: {e}");
            return None;
        }
    };

    let mut buf = Vec::new();
    if let Err(e) = bot.download_file(&file.path, &mut buf).await {
        warn!("下載圖片失敗: {e}");
        return None;
    }

    // 由副檔名推斷 MIME type
    let mime = if file.path.ends_with(".png") {
        "image/png"
    } else {
        "image/jpeg"
    };

    // 縮放 + 格式轉換
    let (mime, buf) = super::process_image(mime, buf);

    Some(ImageData {
        mime_type: mime,
        data: buf,
        file_id: Some(largest.file.id.to_string()),
        source_chat_id: Some(msg.chat.id.0),
        source_message_id: Some(msg.id.0),
    })
}

/// 判斷訊息是否包含需要處理的內容（文字或圖片）
fn has_content(msg: &Message) -> bool {
    msg.text().is_some() || msg.caption().is_some() || msg.photo().is_some()
}

/// Telegram 主動通知器（動態讀取已知 chat_id）
pub struct TelegramNotifier {
    bot: Bot,
    known_chat_ids: KnownChatIds,
}

impl Notifier for TelegramNotifier {
    fn send<'a>(&'a self, message: &'a str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let chat_ids: Vec<ChatId> = self
                .known_chat_ids
                .lock()
                .unwrap()
                .iter()
                .map(|&id| ChatId(id))
                .collect();
            for chat_id in chat_ids {
                send_reply(&self.bot, chat_id, message).await;
            }
            Ok(())
        })
    }
}

impl Channel for TelegramChannel {
    fn notifier(&self) -> Option<Box<dyn Notifier>> {
        Some(Box::new(TelegramNotifier {
            bot: self.bot.clone(),
            known_chat_ids: Arc::clone(&self.known_chat_ids),
        }))
    }

    fn start(
        self: Box<Self>,
        handler: MessageHandler,
        commands: Arc<CommandRegistry>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> {
        Box::pin(async move {
            let bot = self.bot;
            let allowed_users = self.allowed_users;
            let known_chat_ids = self.known_chat_ids;
            let active_task: ActiveTask = Arc::new(std::sync::Mutex::new(None));

            // 將已註冊的指令推送到 Telegram 介面
            let bot_commands: Vec<BotCommand> = commands
                .command_list()
                .into_iter()
                .map(|(name, desc)| BotCommand::new(name, desc))
                .collect();
            if let Err(e) = bot.set_my_commands(bot_commands).await {
                warn!("設定 Telegram 指令選單失敗: {e}");
            }

            info!("Telegram Bot 開始監聽");

            Dispatcher::builder(
                bot,
                Update::filter_message()
                    .filter(|msg: Message| has_content(&msg))
                    .endpoint(move |bot: Bot, msg: Message| {
                        let handler = Arc::clone(&handler);
                        let allowed = Arc::clone(&allowed_users);
                        let commands = Arc::clone(&commands);
                        let active_task = Arc::clone(&active_task);
                        let known_chat_ids = Arc::clone(&known_chat_ids);
                        async move {
                            // 檢查允許名單
                            if !allowed.is_empty() {
                                let user = msg.from.as_ref();
                                let uid =
                                    user.map(|u| u.id.0.to_string()).unwrap_or_default();
                                let uname = user
                                    .and_then(|u| u.username.as_ref())
                                    .map(|n| format!("@{n}"));

                                let ok = allowed.contains(&uid)
                                    || uname
                                        .as_ref()
                                        .is_some_and(|n| allowed.contains(n));

                                if !ok {
                                    warn!(uid, ?uname, "未授權的使用者，忽略訊息");
                                    return respond(());
                                }
                            }

                            // 記錄已驗證使用者的 chat_id（供 notifier 使用）
                            known_chat_ids.lock().unwrap().insert(msg.chat.id.0);

                            let sender_id = msg
                                .from
                                .as_ref()
                                .map(|u| u.id.0.to_string())
                                .unwrap_or_default();

                            let text = extract_text(&msg).unwrap_or_default();

                            // /stop：中斷正在處理的任務（在指令註冊表之前攔截）
                            if msg.photo().is_none() {
                                if let Some((cmd, _)) = parse_command(&text) {
                                    if cmd == "stop" {
                                        let taken = active_task.lock().unwrap().take();
                                        if let Some((h, t)) = taken {
                                            h.abort();
                                            t.abort();
                                            info!("使用者中斷了正在處理的任務");
                                            send_reply(&bot, msg.chat.id, "已中斷處理。").await;
                                        } else {
                                            send_reply(&bot, msg.chat.id, "目前沒有正在處理的任務。").await;
                                        }
                                        return respond(());
                                    }
                                }
                            }

                            // 檢查是否為已註冊的指令（僅純文字訊息）
                            if msg.photo().is_none()
                                && let Some((cmd, args)) = parse_command(&text)
                                && let Some(cmd_handler) = commands.resolve(cmd)
                            {
                                let result = cmd_handler(sender_id, args.to_owned()).await;
                                match result {
                                    Ok(reply) => send_reply(&bot, msg.chat.id, &reply).await,
                                    Err(e) => {
                                        let _ = bot
                                            .send_message(msg.chat.id, format!("指令失敗: {e:#}"))
                                            .await;
                                    }
                                }
                                return respond(());
                            }

                            // 下載圖片：當前訊息 + 被回覆訊息的圖片
                            let mut images = Vec::new();
                            // 被回覆訊息的圖片（放前面作為上下文）
                            if let Some(reply) = msg.reply_to_message() {
                                if reply.photo().is_some() {
                                    if let Some(img) = download_photo(&bot, reply).await {
                                        images.push(img);
                                    }
                                }
                            }
                            // 當前訊息的圖片
                            if msg.photo().is_some() {
                                if let Some(img) = download_photo(&bot, &msg).await {
                                    images.push(img);
                                }
                            }

                            // 持續發送「輸入中…」狀態直到處理完成
                            let typing_bot = bot.clone();
                            let typing_chat_id = msg.chat.id;
                            let typing_handle = tokio::spawn(async move {
                                loop {
                                    let _ = typing_bot
                                        .send_chat_action(typing_chat_id, ChatAction::Typing)
                                        .await;
                                    tokio::time::sleep(std::time::Duration::from_secs(4)).await;
                                }
                            });

                            // 建立串流進度通道
                            let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
                            let progress_bot = bot.clone();
                            let progress_chat_id = msg.chat.id;
                            tokio::spawn(async move {
                                while let Some(text) = progress_rx.recv().await {
                                    send_reply(&progress_bot, progress_chat_id, &text).await;
                                }
                            });

                            let incoming = IncomingMessage {
                                bot_id: String::new(),
                                sender_id,
                                text,
                                images,
                                reply_handle: ReplyHandle::Telegram {
                                    chat_id: msg.chat.id.0,
                                },
                                progress: Some(progress_tx),
                            };

                            // 將 handler 作為獨立 task 執行，以便 /stop 可中斷
                            let handler_task = tokio::spawn(async move {
                                handler(incoming).await
                            });

                            // 儲存 abort handles
                            {
                                let mut guard = active_task.lock().unwrap();
                                *guard = Some((handler_task.abort_handle(), typing_handle.abort_handle()));
                            }

                            let result = match handler_task.await {
                                Ok(r) => r,
                                Err(e) if e.is_cancelled() => {
                                    // 被 /stop 中斷
                                    typing_handle.abort();
                                    return respond(());
                                }
                                Err(e) => {
                                    typing_handle.abort();
                                    error!("Handler panicked: {e}");
                                    let _ = bot
                                        .send_message(msg.chat.id, "內部錯誤")
                                        .await;
                                    return respond(());
                                }
                            };

                            typing_handle.abort();

                            // 清除 active task
                            {
                                let mut guard = active_task.lock().unwrap();
                                *guard = None;
                            }

                            match result {
                                Ok(Some(reply)) => {
                                    send_reply(&bot, msg.chat.id, &reply).await;
                                }
                                Ok(None) => {
                                    // 靜默處理：用 emoji 表情回應
                                    let _ = bot
                                        .set_message_reaction(msg.chat.id, msg.id)
                                        .reaction(vec![ReactionType::Emoji {
                                            emoji: "👀".to_owned(),
                                        }])
                                        .await;
                                }
                                Err(e) => {
                                    error!("處理訊息失敗: {e:#}");
                                    let _ = bot
                                        .send_message(msg.chat.id, format!("處理訊息失敗: {e:#}"))
                                        .await;
                                }
                            }

                            respond(())
                        }
                    }),
            )
            .build()
            .dispatch()
            .await;

            Ok(())
        })
    }
}
