use std::sync::Arc;

use anyhow::Result;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt};

use super::{Channel, CommandRegistry, IncomingMessage, MessageHandler, ReplyHandle};

/// CLI 互動式頻道，用於測試 LLM 設定
pub struct CliChannel;

impl Channel for CliChannel {
    fn start(
        self: Box<Self>,
        handler: MessageHandler,
        commands: Arc<CommandRegistry>,
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

                // 檢查是否為已註冊的指令
                if let Some(without_slash) = text.strip_prefix('/') {
                    let (cmd, args) = without_slash
                        .split_once(char::is_whitespace)
                        .unwrap_or((without_slash, ""));

                    if let Some(cmd_handler) = commands.resolve(cmd) {
                        match cmd_handler("cli".into(), args.trim().to_owned()).await {
                            Ok(reply) => {
                                stdout.write_all(reply.as_bytes()).await?;
                                stdout.write_all(b"\n").await?;
                            }
                            Err(e) => {
                                let msg = format!("指令失敗: {e:#}\n");
                                stdout.write_all(msg.as_bytes()).await?;
                            }
                        }
                        stdout.flush().await?;
                        continue;
                    }
                }

                let msg = IncomingMessage {
                    bot_id: "cli".into(),
                    sender_id: "cli".into(),
                    text,
                    images: vec![],
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
