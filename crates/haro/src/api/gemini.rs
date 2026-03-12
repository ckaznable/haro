use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use base64::Engine;

use super::{EmbeddingProvider, GenerateParams, GenerateResult, ImageInput, LlmProvider};
use crate::tool::ToolRegistry;

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const FILES_UPLOAD_URL: &str = "https://generativelanguage.googleapis.com/upload/v1beta/files";
const FILES_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Gemini API Provider（每個實例對應一個 model）
pub struct GeminiProvider {
    http: Client,
    api_key: String,
    model: String,
    dimension: Option<u64>,
    thinking_budget: Option<i32>,
    thinking_level: Option<String>,
}

// ── 共用 Part 型別（支援文字 + 圖片） ──

/// 多模態 Part：text / inline_data / file_data
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ContentPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<InlineData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_data: Option<FileData>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InlineData {
    mime_type: String,
    data: String, // base64
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileData {
    file_uri: String,
    mime_type: String,
}

impl ContentPart {
    fn text(s: impl Into<String>) -> Self {
        Self { text: Some(s.into()), inline_data: None, file_data: None }
    }

    fn image(mime_type: &str, bytes: &[u8]) -> Self {
        Self {
            text: None,
            inline_data: Some(InlineData {
                mime_type: mime_type.to_owned(),
                data: base64::engine::general_purpose::STANDARD.encode(bytes),
            }),
            file_data: None,
        }
    }

    fn file(uri: &str, mime_type: &str) -> Self {
        Self {
            text: None,
            inline_data: None,
            file_data: Some(FileData {
                file_uri: uri.to_owned(),
                mime_type: mime_type.to_owned(),
            }),
        }
    }
}

/// 從 ImageInput 列表建構 ContentPart 列表（text + inline_data，用於 generate）
fn build_multimodal_parts(text: &str, images: &[ImageInput]) -> Vec<ContentPart> {
    let mut parts = Vec::with_capacity(1 + images.len());
    if !text.is_empty() {
        parts.push(ContentPart::text(text));
    }
    for img in images {
        parts.push(ContentPart::image(&img.mime_type, &img.data));
    }
    // 至少要有一個 part（空字串會被 Gemini API 拒絕）
    if parts.is_empty() {
        parts.push(ContentPart::text("(empty)"));
    }
    parts
}

/// 從已上傳的檔案建構 ContentPart 列表（text + file_data，用於 embed）
fn build_file_parts(text: &str, files: &[UploadedFile]) -> Vec<ContentPart> {
    let mut parts = Vec::with_capacity(1 + files.len());
    if !text.is_empty() {
        parts.push(ContentPart::text(text));
    }
    for f in files {
        parts.push(ContentPart::file(&f.uri, &f.mime_type));
    }
    if parts.is_empty() {
        parts.push(ContentPart::text(""));
    }
    parts
}

// ── File API 型別 ──

/// 上傳至 Gemini File API 的檔案資訊
struct UploadedFile {
    name: String,
    uri: String,
    mime_type: String,
}

#[derive(Deserialize)]
struct FileUploadResponse {
    file: FileInfo,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FileInfo {
    name: String,
    uri: String,
    mime_type: String,
}

// ── Embedding API 型別 ──

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedRequest {
    content: EmbedContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u64>,
}

#[derive(Serialize)]
struct EmbedContent {
    parts: Vec<ContentPart>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: EmbeddingValues,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

// ── Batch Embedding API 型別 ──

#[derive(Serialize)]
struct BatchEmbedRequest {
    requests: Vec<BatchEmbedItem>,
}

#[derive(Serialize)]
struct BatchEmbedItem {
    model: String,
    content: EmbedContent,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<EmbeddingValues>,
}

// ── Generate API 型別 ──

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerateRequest {
    contents: [MultimodalMessage; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<SystemInstruction>,
    generation_config: GenConfig,
}

#[derive(Serialize)]
struct MultimodalMessage {
    role: &'static str,
    parts: Vec<ContentPart>,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<ContentPart>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<&'static str>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<ThinkingConfigPayload>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ThinkingConfigPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_level: Option<String>,
}

// ── Response 型別 ──

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    #[serde(default)]
    prompt_token_count: i32,
    #[serde(default)]
    candidates_token_count: i32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateResponse {
    candidates: Vec<Candidate>,
    #[serde(default)]
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize)]
struct Candidate {
    content: CandidateContent,
}

#[derive(Deserialize)]
struct CandidateContent {
    parts: Vec<ResponsePart>,
}

#[derive(Deserialize)]
struct ResponsePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thought: Option<bool>,
}

// ── Tool Calling 型別 ──

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolGenerateRequest<'a> {
    contents: &'a [ToolMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<&'a ToolInstruction>,
    generation_config: GenConfig,
    tools: [ToolSpec<'a>; 1],
}

#[derive(Serialize)]
struct ToolMessage {
    role: String,
    parts: Vec<ToolRequestPart>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolRequestPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<InlineData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<FnCallData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<FnResponseData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought_signature: Option<String>,
}

impl ToolRequestPart {
    fn text(s: String) -> Self {
        Self { text: Some(s), inline_data: None, function_call: None, function_response: None, thought: None, thought_signature: None }
    }
    fn image(mime_type: &str, bytes: &[u8]) -> Self {
        Self {
            text: None,
            inline_data: Some(InlineData {
                mime_type: mime_type.to_owned(),
                data: base64::engine::general_purpose::STANDARD.encode(bytes),
            }),
            function_call: None,
            function_response: None,
            thought: None,
            thought_signature: None,
        }
    }
    fn function_call(fc: FnCallData) -> Self {
        Self { text: None, inline_data: None, function_call: Some(fc), function_response: None, thought: None, thought_signature: None }
    }
    fn function_response(fr: FnResponseData) -> Self {
        Self { text: None, inline_data: None, function_call: None, function_response: Some(fr), thought: None, thought_signature: None }
    }
}

#[derive(Serialize)]
struct ToolInstruction {
    parts: [ToolTextPart; 1],
}

#[derive(Serialize)]
struct ToolTextPart {
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolSpec<'a> {
    function_declarations: &'a [FnDecl],
}

#[derive(Serialize)]
struct FnDecl {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone)]
struct FnCallData {
    name: String,
    args: serde_json::Value,
}

#[derive(Serialize)]
struct FnResponseData {
    name: String,
    response: serde_json::Value,
}

// ── Tool Calling Response 型別 ──

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolGenerateResponse {
    candidates: Vec<ToolCandidate>,
    #[serde(default)]
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize)]
struct ToolCandidate {
    content: ToolCandidateContent,
}

#[derive(Deserialize)]
struct ToolCandidateContent {
    parts: Vec<ToolResponsePart>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolResponsePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    function_call: Option<FnCallData>,
    #[serde(default)]
    thought: Option<bool>,
    #[serde(default)]
    thought_signature: Option<String>,
}

// ── 實作 ──

impl GeminiProvider {
    pub fn new(
        api_key: &str,
        model: &str,
        dimension: Option<u64>,
        thinking_budget: Option<i32>,
        thinking_level: Option<&str>,
    ) -> Self {
        Self {
            http: Client::new(),
            api_key: api_key.to_owned(),
            model: model.to_owned(),
            dimension,
            thinking_budget,
            thinking_level: thinking_level.map(str::to_owned),
        }
    }

    fn build_thinking_config(&self) -> Option<ThinkingConfigPayload> {
        if self.thinking_budget.is_some() || self.thinking_level.is_some() {
            Some(ThinkingConfigPayload {
                thinking_budget: self.thinking_budget,
                thinking_level: self.thinking_level.clone(),
            })
        } else {
            None
        }
    }

    /// 上傳檔案到 Gemini File API
    async fn upload_file(&self, mime_type: &str, data: &[u8]) -> Result<UploadedFile> {
        let url = format!("{}?key={}", FILES_UPLOAD_URL, self.api_key);

        let resp = self
            .http
            .post(&url)
            .header("X-Goog-Upload-Protocol", "raw")
            .header("Content-Type", mime_type)
            .body(data.to_vec())
            .send()
            .await
            .context("File API 上傳失敗")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("File API 上傳錯誤 {status}: {body}");
        }

        let result: FileUploadResponse = resp.json().await.context("解析 File API 回應失敗")?;
        Ok(UploadedFile {
            name: result.file.name,
            uri: result.file.uri,
            mime_type: result.file.mime_type,
        })
    }

    /// 刪除已上傳的檔案
    async fn delete_file(&self, name: &str) {
        let url = format!("{}/{}?key={}", FILES_BASE_URL, name, self.api_key);
        let _ = self.http.delete(&url).send().await;
    }

    /// 上傳多張圖片，回傳已上傳的檔案列表
    async fn upload_images(&self, images: &[ImageInput]) -> Vec<UploadedFile> {
        let mut files = Vec::with_capacity(images.len());
        for img in images {
            match self.upload_file(&img.mime_type, &img.data).await {
                Ok(f) => files.push(f),
                Err(e) => tracing::warn!("圖片上傳失敗，跳過: {e:#}"),
            }
        }
        files
    }

    /// 清理已上傳的檔案
    async fn cleanup_files(&self, files: Vec<UploadedFile>) {
        for f in &files {
            self.delete_file(&f.name).await;
        }
    }
}

impl EmbeddingProvider for GeminiProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_multimodal(text, &[]).await
    }

    async fn embed_multimodal(&self, text: &str, images: &[ImageInput]) -> Result<Vec<f32>> {
        // 有圖片時使用 File API 上傳，用 file_data 引用
        let uploaded = if !images.is_empty() {
            self.upload_images(images).await
        } else {
            vec![]
        };

        let parts = if uploaded.is_empty() {
            build_multimodal_parts(text, &[])
        } else {
            build_file_parts(text, &uploaded)
        };

        let url = format!(
            "{}/{}:embedContent?key={}",
            BASE_URL, self.model, self.api_key
        );

        let body = EmbedRequest {
            content: EmbedContent { parts },
            output_dimensionality: self.dimension,
        };

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Embedding API 請求失敗")?;

        // 無論成功失敗都清理上傳的檔案
        let cleanup = !uploaded.is_empty();
        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            if cleanup { self.cleanup_files(uploaded).await; }
            anyhow::bail!("Embedding API 錯誤 {status}: {body_text}");
        }

        let result: EmbedResponse = resp.json().await.context("解析 Embedding 回應失敗")?;
        if cleanup { self.cleanup_files(uploaded).await; }
        Ok(result.embedding.values)
    }

    async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = format!(
            "{}/{}:batchEmbedContents?key={}",
            BASE_URL, self.model, self.api_key
        );

        let model_ref = format!("models/{}", self.model);
        let requests = texts
            .iter()
            .map(|t| BatchEmbedItem {
                model: model_ref.clone(),
                content: EmbedContent {
                    parts: vec![ContentPart::text(t)],
                },
            })
            .collect();

        let body = BatchEmbedRequest { requests };

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Batch Embedding API 請求失敗")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Batch Embedding API 錯誤 {status}: {body}");
        }

        let result: BatchEmbedResponse = resp
            .json()
            .await
            .context("解析 Batch Embedding 回應失敗")?;

        Ok(result.embeddings.into_iter().map(|e| e.values).collect())
    }
}

impl LlmProvider for GeminiProvider {
    async fn count_tokens(&self, text: &str) -> Result<i32> {
        let url = format!(
            "{}/{}:countTokens?key={}",
            BASE_URL, self.model, self.api_key
        );

        let body = serde_json::json!({
            "contents": [{"parts": [{"text": text}]}]
        });

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("countTokens API 請求失敗")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("countTokens API 錯誤 {status}: {body}");
        }

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct CountResponse {
            total_tokens: i32,
        }

        let result: CountResponse = resp.json().await.context("解析 countTokens 回應失敗")?;
        Ok(result.total_tokens)
    }

    async fn generate(&self, params: GenerateParams<'_>) -> Result<GenerateResult> {
        let url = format!(
            "{}/{}:generateContent?key={}",
            BASE_URL, self.model, self.api_key
        );

        let user_parts = build_multimodal_parts(params.user_message, params.images);

        let body = GenerateRequest {
            contents: [MultimodalMessage {
                role: "user",
                parts: user_parts,
            }],
            system_instruction: params.system.map(|s| SystemInstruction {
                parts: vec![ContentPart::text(s)],
            }),
            generation_config: GenConfig {
                response_mime_type: if params.json_mode {
                    Some("application/json")
                } else {
                    None
                },
                temperature: params.temperature,
                thinking_config: self.build_thinking_config(),
            },
        };

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("LLM API 請求失敗")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM API 錯誤 {status}: {body}");
        }

        let result: GenerateResponse = resp.json().await.context("解析 LLM 回應失敗")?;

        let usage = result.usage_metadata.unwrap_or_default();
        let text = result
            .candidates
            .into_iter()
            .next()
            .context("LLM 回應中沒有 candidate")?
            .content
            .parts
            .into_iter()
            .filter(|p| !p.thought.unwrap_or(false))
            .filter_map(|p| p.text)
            .next()
            .context("LLM 回應中沒有有效內容")?;

        Ok(GenerateResult {
            text,
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
        })
    }

    async fn generate_with_tools(
        &self,
        params: GenerateParams<'_>,
        tools: &ToolRegistry,
        progress: Option<&super::ProgressSender>,
    ) -> Result<GenerateResult> {
        let defs = tools.definitions();
        if defs.is_empty() {
            return self.generate(params).await;
        }

        let url = format!(
            "{}/{}:streamGenerateContent?key={}&alt=sse",
            BASE_URL, self.model, self.api_key
        );

        let fn_decls: Vec<FnDecl> = defs
            .into_iter()
            .map(|d| FnDecl {
                name: d.name,
                description: d.description,
                parameters: d.parameters,
            })
            .collect();

        let instruction = params.system.map(|s| ToolInstruction {
            parts: [ToolTextPart {
                text: s.to_owned(),
            }],
        });

        let mut user_parts = vec![ToolRequestPart::text(params.user_message.to_owned())];
        for img in params.images {
            user_parts.push(ToolRequestPart::image(&img.mime_type, &img.data));
        }

        let mut contents = vec![ToolMessage {
            role: "user".into(),
            parts: user_parts,
        }];

        const MAX_ROUNDS: usize = 5;
        let mut total_input = 0i32;
        let mut total_output = 0i32;

        for _ in 0..MAX_ROUNDS {
            let body = ToolGenerateRequest {
                contents: &contents,
                system_instruction: instruction.as_ref(),
                generation_config: GenConfig {
                    response_mime_type: None,
                    temperature: params.temperature,
                    thinking_config: self.build_thinking_config(),
                },
                tools: [ToolSpec {
                    function_declarations: &fn_decls,
                }],
            };

            let resp = self
                .http
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("Tool LLM API 請求失敗")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("Tool LLM API 錯誤 {status}: {text}");
            }

            // 解析 SSE 串流，累積所有 parts
            let (parts, usage) = parse_sse_tool_stream(resp).await?;

            total_input += usage.prompt_token_count;
            total_output += usage.candidates_token_count;

            let mut fn_calls = Vec::new();
            let mut texts = Vec::new();

            for part in &parts {
                let is_thought = part.thought.unwrap_or(false);
                if !is_thought {
                    if let Some(fc) = &part.function_call {
                        fn_calls.push(fc.clone());
                    } else if let Some(text) = &part.text {
                        texts.push(text.clone());
                    }
                }
            }

            if fn_calls.is_empty() {
                let text = texts
                    .into_iter()
                    .next()
                    .context("Tool LLM 回應中沒有文字內容")?;
                return Ok(GenerateResult {
                    text,
                    input_tokens: total_input,
                    output_tokens: total_output,
                });
            }

            // 有工具呼叫：先將累積的文字透過 progress 送出，再送工具描述
            if let Some(tx) = progress {
                let buffered: String = texts.into_iter().collect();
                if !buffered.is_empty() {
                    let _ = tx.send(buffered);
                }
                for fc in &fn_calls {
                    let desc = tools.display_call(&fc.name, &fc.args);
                    let _ = tx.send(desc);
                }
            }

            // 加入模型的所有 parts 到對話歷史（保留 thought + thought_signature）
            contents.push(ToolMessage {
                role: "model".into(),
                parts: parts
                    .into_iter()
                    .map(|p| ToolRequestPart {
                        text: p.text,
                        inline_data: None,
                        function_call: p.function_call,
                        function_response: None,
                        thought: p.thought,
                        thought_signature: p.thought_signature,
                    })
                    .collect(),
            });

            // 執行工具並加入回應
            let mut response_parts = Vec::new();
            for fc in &fn_calls {
                let result = tools.call(&fc.name, fc.args.clone()).await?;
                response_parts.push(ToolRequestPart::function_response(FnResponseData {
                    name: fc.name.clone(),
                    response: serde_json::json!({"result": result}),
                }));
            }
            contents.push(ToolMessage {
                role: "user".into(),
                parts: response_parts,
            });
        }

        anyhow::bail!("工具呼叫超過最大輪數 ({MAX_ROUNDS})")
    }
}

/// 解析 Gemini SSE 串流回應，合併所有 chunk 的 parts 和 usage
async fn parse_sse_tool_stream(
    resp: reqwest::Response,
) -> Result<(Vec<ToolResponsePart>, UsageMetadata)> {
    use futures::StreamExt;

    let mut merged_parts: Vec<ToolResponsePart> = Vec::new();
    let mut usage = UsageMetadata::default();

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("SSE 串流讀取失敗")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        // 處理 buf 中所有完整的 SSE 事件
        while let Some(data_line) = extract_sse_data(&mut buf) {
            let chunk_resp: ToolGenerateResponse = match serde_json::from_str(&data_line) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("SSE chunk 解析失敗，跳過: {e}");
                    continue;
                }
            };

            if let Some(u) = chunk_resp.usage_metadata {
                usage = u;
            }

            for candidate in chunk_resp.candidates {
                for part in candidate.content.parts {
                    merge_sse_part(&mut merged_parts, part);
                }
            }
        }
    }

    Ok((merged_parts, usage))
}

/// 從 SSE 緩衝區提取下一個完整的 data: 事件
fn extract_sse_data(buf: &mut String) -> Option<String> {
    // SSE 格式: "data: {...}\n\n"
    let data_prefix = "data: ";
    let start = buf.find(data_prefix)?;
    let content_start = start + data_prefix.len();
    // 找到事件結尾（雙換行）
    let end = buf[content_start..].find("\n\n").or_else(|| buf[content_start..].find("\r\n\r\n"))?;
    let data = buf[content_start..content_start + end].to_owned();
    buf.drain(..content_start + end + 2); // +2 for \n\n
    Some(data)
}

/// 合併 SSE chunk 的 part 到累積列表
/// 純文字 part 會追加到最後一個同類 part，其他 part（function_call 等）直接推入
fn merge_sse_part(parts: &mut Vec<ToolResponsePart>, part: ToolResponsePart) {
    let is_thought = part.thought.unwrap_or(false);

    // function_call / thought_signature 等結構性 part 直接追加
    if part.function_call.is_some() || part.thought_signature.is_some() {
        parts.push(part);
        return;
    }

    // 文字 part：嘗試追加到最後一個同類（thought 屬性相同）的文字 part
    if let Some(text) = &part.text {
        if let Some(last) = parts.last_mut() {
            let last_is_thought = last.thought.unwrap_or(false);
            if last.text.is_some() && last.function_call.is_none() && last_is_thought == is_thought {
                last.text.as_mut().unwrap().push_str(text);
                return;
            }
        }
    }

    parts.push(part);
}
