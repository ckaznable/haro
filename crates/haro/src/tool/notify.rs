use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use serde_json::json;

use crate::channel::Notifier;

use super::{Tool, ToolDef};

/// 建立 send_message 工具
pub fn tool(notifiers: Vec<Arc<dyn Notifier>>) -> Box<dyn Tool> {
    Box::new(SendMessageTool { notifiers })
}

struct SendMessageTool {
    notifiers: Vec<Arc<dyn Notifier>>,
}

impl Tool for SendMessageTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "send_message".into(),
            description: "Send a message to the agent's channels (e.g. Telegram). \
                Use this to proactively notify users about important events, \
                scheduled reminders, or task results."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message text to send"
                    }
                },
                "required": ["message"]
            }),
        }
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            if message.is_empty() {
                return Ok("錯誤：未提供訊息內容".into());
            }

            let mut sent = 0;
            let mut errors = 0;
            for notifier in &self.notifiers {
                if let Err(e) = notifier.send(message).await {
                    tracing::error!("通知發送失敗: {e:#}");
                    errors += 1;
                } else {
                    sent += 1;
                }
            }

            if sent == 0 && errors == 0 {
                Ok("警告：目前沒有已知的聊天對象，訊息未送出。需要使用者先發送訊息給 Bot 才能建立連線。".into())
            } else if sent == 0 {
                Ok(format!("發送失敗（{errors} 個頻道出錯）"))
            } else {
                Ok(format!("已發送到 {sent} 個頻道"))
            }
        })
    }
}
