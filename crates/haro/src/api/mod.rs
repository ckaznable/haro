pub mod gemini;

use anyhow::Result;

use serde::Deserialize;

use crate::models::DistilledData;
use crate::tool::ToolRegistry;

// ── Provider 抽象層 ──

/// 圖片輸入（跨 provider 共用）
pub struct ImageInput {
    pub mime_type: String,
    pub data: Vec<u8>,
}

/// Embedding 服務提供者抽象（可獨立於 LLM 替換，例如本地模型 / Cohere / Voyage）
pub trait EmbeddingProvider: Send + Sync {
    /// 文本 → 向量
    fn embed(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Vec<f32>>> + Send;

    /// 多模態 → 向量（文字 + 圖片）
    fn embed_multimodal(
        &self,
        text: &str,
        images: &[ImageInput],
    ) -> impl std::future::Future<Output = Result<Vec<f32>>> + Send;

    /// 批次文本 → 批次向量（provider 可實作原生 batch API）
    fn batch_embed(
        &self,
        texts: &[String],
    ) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send;
}

/// 通用生成參數
pub struct GenerateParams<'a> {
    pub system: Option<&'a str>,
    pub user_message: &'a str,
    /// 附帶的圖片
    pub images: &'a [ImageInput],
    pub json_mode: bool,
    pub temperature: f32,
}

/// LLM 生成結果（含 token 用量）
pub struct GenerateResult {
    pub text: String,
    pub input_tokens: i32,
    pub output_tokens: i32,
}

/// LLM 服務提供者抽象（OpenAI / Anthropic / Gemini / 本地模型）
pub trait LlmProvider: Send + Sync {
    /// 計算文本的 token 數量
    fn count_tokens(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<i32>> + Send;

    /// 通用文本生成（支援多模態輸入）
    fn generate(
        &self,
        params: GenerateParams<'_>,
    ) -> impl std::future::Future<Output = Result<GenerateResult>> + Send;

    /// 帶工具呼叫的文本生成（多輪 function calling）
    fn generate_with_tools(
        &self,
        params: GenerateParams<'_>,
        tools: &ToolRegistry,
    ) -> impl std::future::Future<Output = Result<GenerateResult>> + Send;
}

// ── 基於 LlmProvider 的上層便利函式 ──

const DISTILL_TEXT_PROMPT: &str = "\
Summarize the following text into JSON for a RAG vector database.

Output format (strict JSON, no markdown fences):
{\"dense_summary\": \"...\", \"keywords\": [\"...\"], \"original_text\": \"...\"}

Rules:
1. `dense_summary`: Ultra-condensed. Strip all filler words, temporal adverbs, and hedging language. \
Retain ONLY core technical entities, causal relationships, and factual claims.
2. `keywords`: Extract proper nouns, protocols, tool/library names, and domain-specific terms only (3-10 items). \
If a term has a well-known Chinese translation, include BOTH forms (e.g. \"Ring Buffer\", \"環狀緩衝區\").
3. `original_text`: The user's input text, reproduced verbatim with zero modifications.";

const DISTILL_IMAGE_PROMPT: &str = "\
Analyze the provided image(s) and any accompanying text, then produce a JSON summary for a RAG vector database.

Output format (strict JSON, no markdown fences):
{\"dense_summary\": \"...\", \"keywords\": [\"...\"], \"original_text\": \"...\"}

Rules:
1. `dense_summary`: Describe what is in the image concisely. Include visible text (OCR), objects, diagrams, \
code snippets, charts, or any identifiable content. If there is accompanying text, integrate it.
2. `keywords`: Extract proper nouns, technical terms, visible labels, brand names, and domain-specific terms (3-10 items). \
If a term has a well-known Chinese translation, include BOTH forms.
3. `original_text`: If the user provided text alongside the image, reproduce it verbatim. \
If there was no text, provide a brief one-line description of the image content.";

/// 容錯解析蒸餾 JSON：去除 markdown fences，忽略尾部多餘字元
fn parse_distilled_json(raw: &str) -> serde_json::Result<DistilledData> {
    let s = raw.trim();
    // 去除 markdown code fences（模型偶爾會包裹）
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s).trim();
    // 使用串流解析器：解析第一個完整 JSON 物件，忽略尾部多餘內容
    let mut de = serde_json::Deserializer::from_str(s);
    DistilledData::deserialize(&mut de)
}

/// 資料蒸餾：將原始文本（可選圖片）轉換為結構化數據，回傳 (蒸餾結果, token 用量)
/// JSON 解析失敗時最多重試 3 次，整體超過 60 秒視為失敗，全部失敗回傳 None
pub async fn distill(
    llm: &impl LlmProvider,
    text: &str,
    images: &[ImageInput],
) -> Result<Option<(DistilledData, GenerateResult)>> {
    match tokio::time::timeout(
        std::time::Duration::from_secs(60),
        distill_inner(llm, text, images),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => {
            tracing::error!("蒸餾超時（60s），跳過向量入庫");
            Ok(None)
        }
    }
}

async fn distill_inner(
    llm: &impl LlmProvider,
    text: &str,
    images: &[ImageInput],
) -> Result<Option<(DistilledData, GenerateResult)>> {
    const MAX_RETRIES: usize = 3;

    let prompt = if images.is_empty() {
        DISTILL_TEXT_PROMPT
    } else {
        DISTILL_IMAGE_PROMPT
    };

    // 圖片訊息如果沒有文字，用佔位符
    let user_message = if text.is_empty() && !images.is_empty() {
        "[圖片]"
    } else {
        text
    };

    for i in 0..MAX_RETRIES {
        let result = llm
            .generate(GenerateParams {
                system: Some(prompt),
                user_message,
                images,
                json_mode: true,
                temperature: 0.1,
            })
            .await?;

        match parse_distilled_json(&result.text) {
            Ok(data) => return Ok(Some((data, result))),
            Err(e) => {
                tracing::warn!("蒸餾 JSON 解析失敗 ({}/{}): {e}", i + 1, MAX_RETRIES);
            }
        }
    }

    tracing::error!("蒸餾重試 {MAX_RETRIES} 次仍失敗，跳過向量入庫");
    Ok(None)
}

/// 生成最終對話回答
pub async fn chat(
    llm: &impl LlmProvider,
    system: &str,
    user_query: &str,
) -> Result<GenerateResult> {
    llm.generate(GenerateParams {
        system: Some(system),
        user_message: user_query,
        images: &[],
        json_mode: false,
        temperature: 0.7,
    })
    .await
}

/// 帶工具呼叫的對話回答
pub async fn chat_with_tools(
    llm: &impl LlmProvider,
    system: &str,
    user_query: &str,
    tools: &ToolRegistry,
) -> Result<GenerateResult> {
    llm.generate_with_tools(
        GenerateParams {
            system: Some(system),
            user_message: user_query,
            images: &[],
            json_mode: false,
            temperature: 0.7,
        },
        tools,
    )
    .await
}

/// 帶圖片的對話回答（含工具）
pub async fn chat_with_images(
    llm: &impl LlmProvider,
    system: &str,
    user_query: &str,
    images: &[ImageInput],
    tools: &ToolRegistry,
) -> Result<GenerateResult> {
    llm.generate_with_tools(
        GenerateParams {
            system: Some(system),
            user_message: user_query,
            images,
            json_mode: false,
            temperature: 0.7,
        },
        tools,
    )
    .await
}
