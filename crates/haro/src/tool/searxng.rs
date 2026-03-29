use std::future::Future;
use std::pin::Pin;

use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use super::{Tool, ToolDef};

pub fn tool(base_url: String) -> Box<dyn Tool> {
    Box::new(SearxngTool {
        http: Client::new(),
        base_url,
    })
}

struct SearxngTool {
    http: Client,
    base_url: String,
}

#[derive(Deserialize)]
struct SearxngResponse {
    #[serde(default)]
    results: Vec<SearxngResult>,
}

#[derive(Deserialize)]
struct SearxngResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
}

impl Tool for SearxngTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "web_search".into(),
            description: "Search the web using SearXNG metasearch engine. \
                Returns search results with title, URL, and snippet for the given query."
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
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("...");
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

            let base = self.base_url.trim_end_matches('/');
            let mut url = reqwest::Url::parse(&format!("{base}/search"))?;
            url.query_pairs_mut()
                .append_pair("q", query)
                .append_pair("format", "json");

            let resp = self.http.get(url).send().await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("SearXNG 搜尋錯誤 {status}: {body}");
            }

            let data: SearxngResponse = resp.json().await?;

            if data.results.is_empty() {
                return Ok("沒有找到相關結果。".into());
            }

            let output: String = data
                .results
                .iter()
                .take(5)
                .enumerate()
                .map(|(i, r)| format!("[{}] {}\n{}\n{}", i + 1, r.title, r.url, r.content))
                .collect::<Vec<_>>()
                .join("\n\n");

            Ok(output)
        })
    }
}
