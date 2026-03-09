use std::future::Future;
use std::pin::Pin;

use anyhow::Result;
use reqwest::Client;
use serde_json::json;

use super::{Tool, ToolDef};

/// 回應內容最大字元數
const MAX_RESPONSE_CHARS: usize = 4000;

pub fn tool() -> Box<dyn Tool> {
    Box::new(FetchTool {
        http: Client::new(),
    })
}

struct FetchTool {
    http: Client,
}

impl Tool for FetchTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "fetch".into(),
            description: "Fetch the content of a URL via HTTP GET. \
                Returns the response body as plain text (truncated to ~4000 chars). \
                Useful for retrieving web pages, API responses, or any publicly accessible URL."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let url = args
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            if url.is_empty() {
                return Ok("錯誤：未提供 URL".into());
            }

            let resp = self
                .http
                .get(url)
                .header("User-Agent", "Haro/1.0")
                .timeout(std::time::Duration::from_secs(10))
                .send()
                .await;

            let resp = match resp {
                Ok(r) => r,
                Err(e) => return Ok(format!("請求失敗：{e}")),
            };

            let status = resp.status();
            let body = match resp.text().await {
                Ok(t) => t,
                Err(e) => return Ok(format!("讀取回應失敗：{e}")),
            };

            let truncated: String = body.chars().take(MAX_RESPONSE_CHARS).collect();
            let suffix = if body.chars().count() > MAX_RESPONSE_CHARS {
                "\n...(已截斷)"
            } else {
                ""
            };

            Ok(format!("[HTTP {status}]\n{truncated}{suffix}"))
        })
    }
}
