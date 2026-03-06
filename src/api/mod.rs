pub mod gemini;

use anyhow::Result;

use crate::models::DistilledData;
use crate::tool::ToolRegistry;

// ── Provider 抽象層 ──

/// Embedding 服務提供者抽象（可獨立於 LLM 替換，例如本地模型 / Cohere / Voyage）
pub trait EmbeddingProvider: Send + Sync {
    /// 文本 → 向量
    fn embed(
        &self,
        text: &str,
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

    /// 通用文本生成
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

const DISTILL_PROMPT: &str = "\
Summarize the following text into JSON for a RAG vector database.

Output format (strict JSON, no markdown fences):
{\"dense_summary\": \"...\", \"keywords\": [\"...\"], \"original_text\": \"...\"}

Rules:
1. `dense_summary`: Ultra-condensed. Strip all filler words, temporal adverbs, and hedging language. \
Retain ONLY core technical entities, causal relationships, and factual claims.
2. `keywords`: Extract proper nouns, protocols, tool/library names, and domain-specific terms only (3-10 items). \
If a term has a well-known Chinese translation, include BOTH forms (e.g. \"Ring Buffer\", \"環狀緩衝區\").
3. `original_text`: The user's input text, reproduced verbatim with zero modifications.";

/// 資料蒸餾：將原始文本轉換為結構化數據，回傳 (蒸餾結果, token 用量)
/// JSON 解析失敗時最多重試 3 次，全部失敗回傳 None
pub async fn distill(
    llm: &impl LlmProvider,
    text: &str,
) -> Result<Option<(DistilledData, GenerateResult)>> {
    const MAX_RETRIES: usize = 3;

    for i in 0..MAX_RETRIES {
        let result = llm
            .generate(GenerateParams {
                system: Some(DISTILL_PROMPT),
                user_message: text,
                json_mode: true,
                temperature: 0.1,
            })
            .await?;

        match serde_json::from_str(result.text.trim()) {
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
            json_mode: false,
            temperature: 0.7,
        },
        tools,
    )
    .await
}
