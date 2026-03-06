use anyhow::Result;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt};

use super::{Channel, IncomingMessage, MessageHandler, ReplyHandle};

/// CLI 互動式頻道，用於測試 LLM 設定
pub struct CliChannel;

impl Channel for CliChannel {
    fn start(
        self: Box<Self>,
        handler: MessageHandler,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> {
        Box::pin(async move {
            let mut stdout = io::stdout();
            let stdin = io::BufReader::new(io::stdin());
            let mut lines = stdin.lines();

            loop {
                stdout.write_all(b"> ").await?;
                stdout.flush().await?;

                let line = match lines.next_line().await? {
                    Some(l) => l,
                    None => break,
                };

                let text = line.trim().to_owned();
                if text.is_empty() {
                    continue;
                }
                if text == "/quit" || text == "/exit" {
                    break;
                }

                let msg = IncomingMessage {
                    bot_id: "cli".into(),
                    sender_id: "cli".into(),
                    text,
                    reply_handle: ReplyHandle::Cli,
                };

                match handler(msg).await {
                    Ok(reply) => {
                        stdout.write_all(reply.as_bytes()).await?;
                        stdout.write_all(b"\n").await?;
                        stdout.flush().await?;
                    }
                    Err(e) => {
                        let msg = format!("錯誤: {e:#}\n");
                        stdout.write_all(msg.as_bytes()).await?;
                        stdout.flush().await?;
                    }
                }
            }

            Ok(())
        })
    }
}
