use std::collections::HashMap;

use anyhow::Result;
use chrono::{DateTime, Utc};
use qdrant_client::Qdrant;
use sqlx::PgPool;
use tracing::info;
use uuid::Uuid;

use crate::api::{self, EmbeddingProvider, GenerateResult, LlmProvider};
use crate::db;
use crate::models::{RankedResult, SearchHit};

/// RRF 常數 k（標準值 60）
const RRF_K: f64 = 60.0;

/// 時間衰減係數 λ（half-life ≈ 69 天）
const TIME_DECAY_LAMBDA: f64 = 0.01;

/// 短訊息門檻（字元數）
const SHORT_MSG_CHARS: usize = 20;

const MEMORIZE_PROMPT: &str = "\
Judge whether the following user message contains meaningful information worth storing as a long-term memory \
(e.g. facts, preferences, decisions, questions with context, instructions). \
Trivial messages like greetings, acknowledgements, single-word replies, or chit-chat should NOT be stored.\n\n\
Reply with ONLY \"yes\" or \"no\".";

// ── 非同步入庫流程 ──

/// 判斷短訊息是否值得存入長期記憶
async fn should_memorize(worker: &impl LlmProvider, text: &str) -> Result<bool> {
    let result = worker
        .generate(api::GenerateParams {
            system: Some(MEMORIZE_PROMPT),
            user_message: text,
            json_mode: false,
            temperature: 0.0,
        })
        .await?;

    let answer = result.text.trim().to_lowercase();
    Ok(answer.starts_with("yes"))
}

/// 入庫流程：短訊息先經 worker 判斷是否值得存入 Qdrant
/// 回傳 (UUID, 累計 worker token 用量)
pub async fn ingest(
    pg: &PgPool,
    qdrant: &Qdrant,
    embedder: &impl EmbeddingProvider,
    worker: &impl LlmProvider,
    bot_id: &str,
    text: &str,
) -> Result<(Uuid, GenerateResult)> {
    // 短訊息先判斷是否值得記憶
    if text.chars().count() <= SHORT_MSG_CHARS {
        let worth = should_memorize(worker, text).await?;
        if !worth {
            let token_count = worker.count_tokens(text).await?;
            let inserted = db::postgres::insert_raw_message(pg, bot_id, text, token_count).await?;
            info!(id = %inserted.id, bot_id, "短訊息不存入 Qdrant");
            return Ok((
                inserted.id,
                GenerateResult {
                    text: String::new(),
                    input_tokens: 0,
                    output_tokens: 0,
                },
            ));
        }
    }

    // 1. Worker 模型資料蒸餾
    let distilled = api::distill(worker, text).await?;

    let Some((data, usage)) = distilled else {
        // 蒸餾失敗，僅存原始訊息到 PG
        let token_count = worker.count_tokens(text).await.unwrap_or(0);
        let inserted = db::postgres::insert_raw_message(pg, bot_id, text, token_count).await?;
        info!(id = %inserted.id, bot_id, "蒸餾失敗，僅存原始訊息");
        return Ok((
            inserted.id,
            GenerateResult {
                text: String::new(),
                input_tokens: 0,
                output_tokens: 0,
            },
        ));
    };

    // 2. 對 dense_summary 計算向量 + 計算原文 token 數
    let (vector, token_count) = tokio::try_join!(
        embedder.embed(&data.dense_summary),
        worker.count_tokens(&data.original_text),
    )?;

    // 3. 寫入 PostgreSQL（取得 PG 產生的 id + created_at）
    let inserted = db::postgres::insert_message(pg, bot_id, &data, token_count).await?;

    // 4. 以相同 UUID + 時間戳寫入 Qdrant
    db::qdrant::upsert_point(
        qdrant,
        inserted.id,
        vector,
        bot_id,
        &data.original_text,
        &data.keywords,
        inserted.created_at.timestamp(),
    )
    .await?;

    info!(id = %inserted.id, bot_id, "記憶入庫完成");
    Ok((inserted.id, usage))
}

// ── 混合搜尋流程 ──

/// 向量 + BM25 並發搜尋 → RRF 融合 → 回傳 top_k 結果
pub async fn retrieve(
    pg: &PgPool,
    qdrant: &Qdrant,
    embedder: &impl EmbeddingProvider,
    bot_id: &str,
    query: &str,
    top_k: usize,
) -> Result<Vec<RankedResult>> {
    // 將問題轉換為向量
    let query_vector = embedder.embed(query).await?;

    // 搜尋數量放大以供 RRF 融合
    let search_limit = (top_k * 3).max(20);

    // 並發雙線搜尋
    let (vector_hits, bm25_hits) = tokio::try_join!(
        db::qdrant::search_vector(qdrant, bot_id, query_vector, search_limit as u64),
        db::postgres::search_bm25(pg, bot_id, query, search_limit as i64),
    )?;

    info!(
        bot_id,
        vector_hits = vector_hits.len(),
        bm25_hits = bm25_hits.len(),
        "混合搜尋完成，進行 RRF 融合"
    );

    Ok(rrf_fuse(&vector_hits, &bm25_hits, top_k))
}

// ── RRF 倒數排名融合 + 時間衰減 ──

fn rrf_fuse(
    vector_hits: &[SearchHit],
    bm25_hits: &[SearchHit],
    top_k: usize,
) -> Vec<RankedResult> {
    // doc_id → (rrf_score_sum, original_text, created_at)
    let mut scores: HashMap<Uuid, (f64, String, DateTime<Utc>)> = HashMap::new();

    // 向量搜尋排名貢獻（已按相似度降序）
    for (rank, hit) in vector_hits.iter().enumerate() {
        let entry = scores
            .entry(hit.id)
            .or_insert_with(|| (0.0, hit.original_text.clone(), hit.created_at));
        entry.0 += 1.0 / (RRF_K + (rank + 1) as f64);
    }

    // BM25 搜尋排名貢獻（已按 ts_rank 降序）
    for (rank, hit) in bm25_hits.iter().enumerate() {
        let entry = scores
            .entry(hit.id)
            .or_insert_with(|| (0.0, hit.original_text.clone(), hit.created_at));
        entry.0 += 1.0 / (RRF_K + (rank + 1) as f64);
    }

    // 套用時間衰減並排序
    let now = Utc::now();
    let mut results: Vec<RankedResult> = scores
        .into_iter()
        .map(|(id, (rrf_score, text, created_at))| {
            let age_days = (now - created_at).num_seconds().max(0) as f64 / 86400.0;
            let decay = (-TIME_DECAY_LAMBDA * age_days).exp();
            RankedResult {
                id,
                score: rrf_score * decay,
                original_text: text,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    results
}
