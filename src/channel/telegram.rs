use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{ChatAction, ParseMode};
use tracing::{error, info, warn};

use super::{Channel, IncomingMessage, MessageHandler, ReplyHandle};

/// Telegram Bot 頻道實作
pub struct TelegramChannel {
    bot: Bot,
    allowed_users: Arc<HashSet<String>>,
}

impl TelegramChannel {
    /// `allowed_users` 可放 user ID（數字）或 @username，空 = 不限制
    pub fn new(bot_token: &str, allowed_users: &[String]) -> Self {
        Self {
            bot: Bot::new(bot_token),
            allowed_users: Arc::new(allowed_users.iter().cloned().collect()),
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

impl Channel for TelegramChannel {
    fn start(
        self: Box<Self>,
        handler: MessageHandler,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> {
        Box::pin(async move {
            let bot = self.bot;
            let allowed_users = self.allowed_users;

            info!("Telegram Bot 開始監聽");

            Dispatcher::builder(
                bot,
                Update::filter_message()
                    .filter_map(|msg: Message| msg.text().map(str::to_owned))
                    .endpoint(move |bot: Bot, msg: Message, text: String| {
                        let handler = Arc::clone(&handler);
                        let allowed = Arc::clone(&allowed_users);
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
                                        .map_or(false, |n| allowed.contains(n));

                                if !ok {
                                    warn!(uid, ?uname, "未授權的使用者，忽略訊息");
                                    return respond(());
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

                            let incoming = IncomingMessage {
                                bot_id: String::new(),
                                sender_id: msg
                                    .from
                                    .as_ref()
                                    .map(|u| u.id.0.to_string())
                                    .unwrap_or_default(),
                                text,
                                reply_handle: ReplyHandle::Telegram {
                                    chat_id: msg.chat.id.0,
                                },
                            };

                            let result = handler(incoming).await;
                            typing_handle.abort();

                            match result {
                                Ok(reply) => {
                                    send_reply(&bot, msg.chat.id, &reply).await;
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
