use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use serde_json::json;
use sqlx::PgPool;

use super::{Tool, ToolDef};
use crate::db::postgres;

/// 約 512 token 的字元上限
const MAX_CHARS: usize = 1500;

/// 回傳備忘錄的寫入與讀取工具
pub fn tools(pg: Arc<PgPool>, bot_id: String) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(SaveMemoTool {
            pg: Arc::clone(&pg),
            bot_id: bot_id.clone(),
        }),
        Box::new(ReadMemoTool { pg, bot_id }),
    ]
}

// ── save_memo ──

struct SaveMemoTool {
    pg: Arc<PgPool>,
    bot_id: String,
}

impl Tool for SaveMemoTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "save_memo".into(),
            description: "Save important information to your short-term scratchpad. \
                The scratchpad holds up to ~512 tokens. New content overwrites old content. \
                Use this to remember critical facts, user preferences, or ongoing task state."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memo content to save (max ~512 tokens)"
                    }
                },
                "required": ["content"]
            }),
        }
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let content = args
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            let content: String = content.chars().take(MAX_CHARS).collect();
            let char_count = content.chars().count();

            postgres::upsert_scratchpad(&self.pg, &self.bot_id, &content).await?;

            Ok(format!("已儲存 {char_count} 字元到備忘錄"))
        })
    }
}

// ── read_memo ──

struct ReadMemoTool {
    pg: Arc<PgPool>,
    bot_id: String,
}

impl Tool for ReadMemoTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "read_memo".into(),
            description: "Read the current content of your short-term scratchpad to confirm \
                what is stored. Returns the full memo content or indicates it is empty."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    fn call(
        &self,
        _args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let content = postgres::get_scratchpad(&self.pg, &self.bot_id).await?;

            match content {
                Some(c) if !c.is_empty() => Ok(c),
                _ => Ok("（備忘錄目前為空）".into()),
            }
        })
    }
}
