use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use serde_json::json;

use super::{Tool, ToolDef};
use crate::api::gemini::GeminiProvider;

pub fn tool(provider: Arc<GeminiProvider>) -> Box<dyn Tool> {
    Box::new(SearchTool { provider })
}

struct SearchTool {
    provider: Arc<GeminiProvider>,
}

impl Tool for SearchTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "google_search".into(),
            description: "Search the web using Google Search. \
                Returns search results for the given query. \
                Use this when you need up-to-date information from the internet."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    fn display_call(&self, args: &serde_json::Value) -> Option<String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("...");
        Some(format!("搜尋 {query}"))
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            if query.is_empty() {
                anyhow::bail!("搜尋查詢不可為空");
            }

            self.provider.grounded_search(query).await
        })
    }
}
