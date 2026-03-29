use std::future::Future;
use std::pin::Pin;

use anyhow::Result;
use regex::Regex;
use reqwest::Client;
use serde_json::json;

use super::{Tool, ToolDef};

/// 回應內容最大字元數
const MAX_RESPONSE_CHARS: usize = 4000;

/// 預設 Accept header（優先 markdown / HTML / JSON）
const DEFAULT_ACCEPT: &str = "text/markdown, text/html, application/json";

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
                HTML responses are automatically cleaned (tag attributes stripped, \
                deeply nested tags flattened) unless clean_html is set to false."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    },
                    "clean_html": {
                        "type": "boolean",
                        "description": "Whether to clean HTML responses (strip tag attributes, flatten deeply nested tags). Default: true"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    fn display_call(&self, args: &serde_json::Value) -> Option<String> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("(unknown)");
        Some(format!("fetch {url}"))
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let url = args.get("url").and_then(|v| v.as_str()).unwrap_or_default();

            if url.is_empty() {
                return Ok("錯誤：未提供 URL".into());
            }

            let clean_html = args
                .get("clean_html")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            let resp = self
                .http
                .get(url)
                .header("User-Agent", "Haro/1.0")
                .header("Accept", DEFAULT_ACCEPT)
                .timeout(std::time::Duration::from_secs(10))
                .send()
                .await;

            let resp = match resp {
                Ok(r) => r,
                Err(e) => return Ok(format!("請求失敗：{e}")),
            };

            let status = resp.status();
            let is_html = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|ct| ct.contains("text/html"));

            let body = match resp.text().await {
                Ok(t) => t,
                Err(e) => return Ok(format!("讀取回應失敗：{e}")),
            };

            let body = if is_html && clean_html {
                clean_html_content(&body)
            } else {
                body
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

/// 清理 HTML：去除標籤屬性，並將深度嵌套的單一文字提升到第一層標籤
pub(crate) fn clean_html_content(html: &str) -> String {
    // 1. 去除所有標籤的屬性：<div class="foo" id="bar"> → <div>
    let attr_re = Regex::new(r"<([a-zA-Z]\w*)\s[^>]*?>").unwrap();
    let mut result = attr_re.replace_all(html, "<$1>").to_string();

    // 2. 逐層扁平化：<a><b>text</b></a> → <a>text</a>
    //    捕獲 outer_open / inner_open / text / inner_close / outer_close 五個部分，
    //    在閉包中驗證標籤配對正確後才合併（避免 backreference）。
    //    只在內容為純文字（[^<]*）時匹配，確保不會跨兄弟節點合併。
    //    反覆套用直到穩定，從最內層往外逐步展開。
    let nest_re =
        Regex::new(r"<([a-zA-Z]\w*)>\s*<([a-zA-Z]\w*)>([^<]*)</([a-zA-Z]\w*)>\s*</([a-zA-Z]\w*)>")
            .unwrap();
    loop {
        let prev = result.clone();
        result = nest_re
            .replace_all(&prev, |caps: &regex::Captures| {
                let outer_open = &caps[1];
                let inner_open = &caps[2];
                let text = &caps[3];
                let inner_close = &caps[4];
                let outer_close = &caps[5];

                if inner_open == inner_close && outer_open == outer_close {
                    format!("<{outer_open}>{text}</{outer_open}>")
                } else {
                    caps[0].to_owned()
                }
            })
            .to_string();
        if result == prev {
            break;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, http::header, routing::get};

    /// 啟動測試用 HTTP server，回傳 base URL
    async fn start_server(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn fetch_tool() -> Box<dyn Tool> {
        tool()
    }

    // ── clean_html_content 單元測試 ──

    #[test]
    fn strip_single_attribute() {
        assert_eq!(
            clean_html_content(r#"<div class="foo">text</div>"#),
            "<div>text</div>",
        );
    }

    #[test]
    fn strip_multiple_attributes() {
        assert_eq!(
            clean_html_content(r#"<div class="a" id="b" style="color:red">text</div>"#),
            "<div>text</div>",
        );
    }

    #[test]
    fn strip_nested_attributes() {
        let html = r#"<div class="outer"><span style="bold"><a href="url">link</a></span></div>"#;
        let cleaned = clean_html_content(html);
        // 屬性去除 + 嵌套扁平化
        assert_eq!(cleaned, "<div>link</div>");
    }

    #[test]
    fn flatten_two_levels() {
        assert_eq!(
            clean_html_content("<div><span>text</span></div>"),
            "<div>text</div>",
        );
    }

    #[test]
    fn flatten_three_levels() {
        assert_eq!(
            clean_html_content("<div><span><a>text</a></span></div>"),
            "<div>text</div>",
        );
    }

    #[test]
    fn no_flatten_siblings() {
        // 有兄弟節點時不應扁平化
        let html = "<div><span>foo</span><span>bar</span></div>";
        let cleaned = clean_html_content(html);
        assert_eq!(cleaned, html);
    }

    #[test]
    fn no_flatten_mixed_content() {
        // 文字 + 子標籤混合時不應扁平化
        let html = "<div>before<span>inner</span>after</div>";
        let cleaned = clean_html_content(html);
        assert_eq!(cleaned, html);
    }

    #[test]
    fn preserve_plain_text() {
        let html = "<p>hello world</p>";
        assert_eq!(clean_html_content(html), html);
    }

    #[test]
    fn closing_tags_untouched() {
        // 確認 closing tag 不受影響
        let html = r#"<div class="x">text</div>"#;
        assert_eq!(clean_html_content(html), "<div>text</div>");
    }

    // ── 整合測試（含 HTTP server） ──

    #[tokio::test]
    async fn fetch_plain_text() {
        let app = Router::new().route("/text", get(|| async { "hello" }));
        let base = start_server(app).await;

        let t = fetch_tool();
        let result = t
            .call(json!({ "url": format!("{base}/text") }))
            .await
            .unwrap();

        assert!(result.contains("[HTTP 200 OK]"));
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn fetch_html_cleaned_by_default() {
        let app = Router::new().route(
            "/page",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                    r#"<div class="wrapper"><span id="inner">content</span></div>"#,
                )
            }),
        );
        let base = start_server(app).await;

        let t = fetch_tool();
        let result = t
            .call(json!({ "url": format!("{base}/page") }))
            .await
            .unwrap();

        // 屬性已去除 + 嵌套已扁平化
        assert!(result.contains("<div>content</div>"));
        assert!(!result.contains("class="));
    }

    #[tokio::test]
    async fn fetch_html_clean_disabled() {
        let html = r#"<div class="keep"><span>text</span></div>"#;
        let app = Router::new().route(
            "/raw",
            get(move || async move { ([(header::CONTENT_TYPE, "text/html")], html) }),
        );
        let base = start_server(app).await;

        let t = fetch_tool();
        let result = t
            .call(json!({ "url": format!("{base}/raw"), "clean_html": false }))
            .await
            .unwrap();

        // clean_html=false：保留原始 HTML
        assert!(result.contains(r#"class="keep""#));
        assert!(result.contains("<span>text</span>"));
    }

    #[tokio::test]
    async fn fetch_json_not_cleaned() {
        let app = Router::new().route(
            "/api",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"key":"<div class=\"x\">val</div>"}"#,
                )
            }),
        );
        let base = start_server(app).await;

        let t = fetch_tool();
        let result = t
            .call(json!({ "url": format!("{base}/api") }))
            .await
            .unwrap();

        // JSON 回應不做 HTML 清理
        assert!(result.contains(r#"class=\"x\""#));
    }

    #[tokio::test]
    async fn fetch_accept_header_sent() {
        use axum::extract::Request;

        let app = Router::new().route(
            "/echo-accept",
            get(|req: Request| async move {
                let accept = req
                    .headers()
                    .get("accept")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("(none)")
                    .to_owned();
                accept
            }),
        );
        let base = start_server(app).await;

        let t = fetch_tool();
        let result = t
            .call(json!({ "url": format!("{base}/echo-accept") }))
            .await
            .unwrap();

        assert!(result.contains("text/markdown"));
        assert!(result.contains("text/html"));
        assert!(result.contains("application/json"));
    }

    #[tokio::test]
    async fn fetch_empty_url_returns_error() {
        let t = fetch_tool();
        let result = t.call(json!({ "url": "" })).await.unwrap();
        assert!(result.contains("未提供 URL"));
    }

    #[tokio::test]
    async fn fetch_truncates_long_response() {
        let long_body = "x".repeat(5000);
        let app = Router::new().route(
            "/long",
            get(move || {
                let body = long_body.clone();
                async move { body }
            }),
        );
        let base = start_server(app).await;

        let t = fetch_tool();
        let result = t
            .call(json!({ "url": format!("{base}/long") }))
            .await
            .unwrap();

        assert!(result.contains("已截斷"));
        // 內容應被截斷到 MAX_RESPONSE_CHARS
        assert!(result.len() < 5000);
    }
}
