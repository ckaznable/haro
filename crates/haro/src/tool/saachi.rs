use std::future::Future;
use std::pin::Pin;

use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use super::{Tool, ToolDef};

pub fn tool(base_url: String) -> Box<dyn Tool> {
    Box::new(SaachiTool {
        http: Client::new(),
        base_url,
    })
}

struct SaachiTool {
    http: Client,
    base_url: String,
}

#[derive(Deserialize)]
struct SaachiResponse {
    #[serde(default)]
    results: Vec<SaachiResult>,
    #[serde(default)]
    errors: Vec<SaachiError>,
}

#[derive(Deserialize)]
struct SaachiResult {
    #[serde(default)]
    engine: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct SaachiError {
    #[serde(default)]
    engine: String,
    #[serde(default)]
    error: String,
}

impl Tool for SaachiTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "saachi_search".into(),
            description: "Search the web using Saachi metasearch service. \
                Returns search results with source engine, title, URL, and snippet for the given query."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "engines": {
                        "type": "string",
                        "description": "Optional comma-separated engines: brave,duckduckgo"
                    },
                    "page": {
                        "type": "integer",
                        "description": "Optional page number from 1 to 10",
                        "minimum": 1,
                        "maximum": 10
                    },
                    "locale": {
                        "type": "string",
                        "description": "Optional locale, for example en-US or zh-TW"
                    },
                    "safe": {
                        "type": "string",
                        "description": "Optional safe search level",
                        "enum": ["off", "moderate", "strict"]
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Optional time range filter",
                        "enum": ["day", "week", "month", "year"]
                    }
                },
                "required": ["query"]
            }),
        }
    }

    fn display_call(&self, args: &serde_json::Value) -> Option<String> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("...");
        Some(format!("Saachi 搜尋 {query}"))
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim();

            if query.is_empty() {
                anyhow::bail!("搜尋查詢不可為空");
            }

            let base = self.base_url.trim_end_matches('/');
            let mut url = reqwest::Url::parse(&format!("{base}/search"))?;
            url.query_pairs_mut().append_pair("q", query);

            if let Some(value) = optional_string(&args, "engines") {
                url.query_pairs_mut().append_pair("engines", value);
            }
            if let Some(page) = optional_page(&args)? {
                url.query_pairs_mut().append_pair("page", &page.to_string());
            }
            if let Some(value) = optional_string(&args, "locale") {
                url.query_pairs_mut().append_pair("locale", value);
            }
            if let Some(value) = optional_string(&args, "safe") {
                url.query_pairs_mut().append_pair("safe", value);
            }
            if let Some(value) = optional_string(&args, "time_range") {
                url.query_pairs_mut().append_pair("time_range", value);
            }

            let resp = self.http.get(url).send().await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("Saachi 搜尋錯誤 {status}: {body}");
            }

            let data: SaachiResponse = resp.json().await?;

            if data.results.is_empty() && data.errors.is_empty() {
                return Ok("沒有找到相關結果。".into());
            }

            let mut sections = Vec::new();
            if !data.results.is_empty() {
                sections.push(format_results(&data.results));
            }
            if !data.errors.is_empty() {
                sections.push(format_errors(&data.errors));
            }

            Ok(sections.join("\n\n"))
        })
    }
}

fn optional_string<'a>(args: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
}

fn optional_page(args: &serde_json::Value) -> Result<Option<u64>> {
    let Some(value) = args.get("page") else {
        return Ok(None);
    };
    let Some(page) = value.as_u64() else {
        anyhow::bail!("page 必須是 1 到 10 的整數");
    };
    if !(1..=10).contains(&page) {
        anyhow::bail!("page 必須是 1 到 10 的整數");
    }
    Ok(Some(page))
}

fn format_results(results: &[SaachiResult]) -> String {
    results
        .iter()
        .take(5)
        .enumerate()
        .map(|(i, r)| {
            let engine = if r.engine.is_empty() {
                "unknown"
            } else {
                &r.engine
            };
            format!(
                "[{}] {}\n{}\n({}) {}",
                i + 1,
                r.title,
                r.url,
                engine,
                r.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn format_errors(errors: &[SaachiError]) -> String {
    let lines = errors
        .iter()
        .map(|e| {
            let engine = if e.engine.is_empty() {
                "unknown"
            } else {
                &e.engine
            };
            format!("- {engine}: {}", e.error)
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("搜尋錯誤:\n{lines}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, extract::Query, routing::get};
    use serde_json::json;
    use std::collections::HashMap;

    async fn start_server(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[test]
    fn tool_name_is_distinct_from_searxng() {
        let t = tool("http://localhost:3000".into());
        assert_eq!(t.definition().name, "saachi_search");
    }

    #[tokio::test]
    async fn search_formats_results_and_errors() {
        let app = Router::new().route(
            "/search",
            get(|Query(params): Query<HashMap<String, String>>| async move {
                assert_eq!(params.get("q"), Some(&"rust axum".to_owned()));
                assert_eq!(params.get("engines"), Some(&"brave,duckduckgo".to_owned()));
                assert_eq!(params.get("page"), Some(&"2".to_owned()));
                Json(json!({
                    "query": "rust axum",
                    "page": 2,
                    "engines": ["brave", "duckduckgo"],
                    "cached": false,
                    "elapsed_ms": 4,
                    "results": [
                        {
                            "engine": "brave",
                            "title": "Rust Axum",
                            "url": "https://example.com/axum",
                            "content": "Axum result",
                            "thumbnail": null
                        }
                    ],
                    "errors": [
                        {"engine": "duckduckgo", "error": "upstream timeout"}
                    ]
                }))
            }),
        );
        let base = start_server(app).await;
        let t = tool(base);

        let result = t
            .call(json!({
                "query": "rust axum",
                "engines": "brave,duckduckgo",
                "page": 2
            }))
            .await
            .unwrap();

        assert!(result.contains("[1] Rust Axum"));
        assert!(result.contains("https://example.com/axum"));
        assert!(result.contains("(brave) Axum result"));
        assert!(result.contains("- duckduckgo: upstream timeout"));
    }

    #[tokio::test]
    async fn rejects_invalid_page() {
        let t = tool("http://localhost:3000".into());
        let result = t.call(json!({"query": "rust", "page": 11})).await;
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("page 必須是 1 到 10")
        );
    }
}
