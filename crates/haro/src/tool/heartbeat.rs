use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use anyhow::Result;
use serde_json::json;

use super::{Tool, ToolDef};

/// 回傳 HEARTBEAT.md 的讀取與編輯工具
pub fn tools(agent_path: PathBuf) -> Vec<Box<dyn Tool>> {
    let hb_path = agent_path.join("HEARTBEAT.md");
    vec![
        Box::new(ReadHeartbeatTool {
            path: hb_path.clone(),
        }),
        Box::new(EditHeartbeatTool { path: hb_path }),
    ]
}

// ── read_heartbeat ──

struct ReadHeartbeatTool {
    path: PathBuf,
}

impl Tool for ReadHeartbeatTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "read_heartbeat".into(),
            description: "Read the current HEARTBEAT.md content. \
                The heartbeat file contains periodic task instructions that are executed \
                automatically on a timer. Use this to check what tasks are currently configured."
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
            match tokio::fs::read_to_string(&self.path).await {
                Ok(content) if !content.is_empty() => Ok(content),
                Ok(_) => Ok("（HEARTBEAT.md 目前為空）".into()),
                Err(_) => Ok("（HEARTBEAT.md 不存在）".into()),
            }
        })
    }
}

// ── edit_heartbeat ──

struct EditHeartbeatTool {
    path: PathBuf,
}

impl Tool for EditHeartbeatTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "edit_heartbeat".into(),
            description: "Overwrite the HEARTBEAT.md file with new content. \
                The heartbeat file contains periodic task instructions that are executed \
                automatically on a timer. Use this to add, modify, or remove scheduled tasks."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The new content for HEARTBEAT.md"
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

            tokio::fs::write(&self.path, content).await?;

            Ok("HEARTBEAT.md 已更新".into())
        })
    }
}
