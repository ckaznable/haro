use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Utc};
use qdrant_client::Qdrant;
use sqlx::PgPool;
use tokio::sync::mpsc;
use tracing::{error, info};
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

// ── 批次入庫（ingest 模式專用） ──

/// 批次入庫佇列的發送端
#[derive(Clone)]
pub struct IngestQueue {
    pg: Arc<PgPool>,
    /// 通知背景任務有新項目（僅作信號，資料在 PG）
    notify: mpsc::Sender<()>,
}

impl IngestQueue {
    /// 將訊息寫入 PG pending_ingest 表後通知背景任務
    pub async fn push(&self, agent_id: &str, text: &str, worker_model: &str) -> Result<()> {
        db::postgres::insert_pending_ingest(&self.pg, agent_id, text, worker_model).await?;
        // 通知背景任務（滿了也沒關係，背景任務會定期 poll PG）
        let _ = self.notify.try_send(());
        Ok(())
    }
}

/// 批次入庫背景任務的設定
pub struct BatchIngestConfig {
    pub pg: Arc<PgPool>,
    pub qdrant: Arc<Qdrant>,
    pub embedder: Arc<crate::api::gemini::GeminiProvider>,
    pub worker: Arc<crate::api::gemini::GeminiProvider>,
    /// 累積多少筆後觸發批次處理
    pub batch_size: usize,
    /// 最長等待時間（超過後即使未滿 batch_size 也處理）
    pub flush_interval: Duration,
}

/// 建立批次入庫佇列與背景任務，回傳 (佇列, JoinHandle)
pub fn spawn_batch_ingest(
    cfg: BatchIngestConfig,
) -> (IngestQueue, tokio::task::JoinHandle<Result<()>>) {
    let (tx, rx) = mpsc::channel::<()>(64);
    let queue = IngestQueue {
        pg: Arc::clone(&cfg.pg),
        notify: tx,
    };
    let handle = tokio::spawn(batch_ingest_loop(rx, cfg));
    (queue, handle)
}

/// 背景 polling 迴圈：收到信號或超時後從 PG 讀取 pending 項目批次處理
async fn batch_ingest_loop(mut rx: mpsc::Receiver<()>, cfg: BatchIngestConfig) -> Result<()> {
    // 啟動時先處理上次中斷遺留的 pending 項目
    let remaining = db::postgres::fetch_pending_ingest(&cfg.pg, cfg.batch_size as i64).await?;
    if !remaining.is_empty() {
        info!(count = remaining.len(), "啟動時發現未處理的 pending 項目，開始處理");
        process_pending_batch(remaining, &cfg).await;
    }

    loop {
        // 等待通知或超時（定期 poll 以防通知遺失）
        let _ = tokio::time::timeout(cfg.flush_interval, rx.recv()).await;

        // 持續等待短暫時間收集更多項目
        let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(())) => continue, // 消耗通知
                _ => break,
            }
        }

        // 從 PG 讀取 pending 項目
        let items = match db::postgres::fetch_pending_ingest(&cfg.pg, cfg.batch_size as i64).await {
            Ok(items) => items,
            Err(e) => {
                error!("讀取 pending_ingest 失敗: {e:#}");
                continue;
            }
        };

        if items.is_empty() {
            continue;
        }

        process_pending_batch(items, &cfg).await;
    }
}

/// 處理一個批次：並行蒸餾 → 批次 embedding → 寫入 PG + Qdrant → 刪除 pending
async fn process_pending_batch(items: Vec<db::postgres::PendingIngestRow>, cfg: &BatchIngestConfig) {
    let count = items.len();
    let pending_ids: Vec<i64> = items.iter().map(|i| i.id).collect();
    info!(count, "開始批次入庫處理");

    // 1. 並行蒸餾（每個 item 獨立 LLM 呼叫）
    let distill_futures: Vec<_> = items
        .iter()
        .map(|item| {
            let worker = Arc::clone(&cfg.worker);
            let text = item.text.clone();
            async move { api::distill(worker.as_ref(), &text).await }
        })
        .collect();

    let distill_results = futures::future::join_all(distill_futures).await;

    // 收集成功蒸餾的項目
    struct DistilledItem<'a> {
        row: &'a db::postgres::PendingIngestRow,
        data: crate::models::DistilledData,
        usage: GenerateResult,
    }
    let mut distilled: Vec<DistilledItem> = Vec::new();

    for (item, result) in items.iter().zip(distill_results) {
        match result {
            Ok(Some((data, usage))) => {
                distilled.push(DistilledItem { row: item, data, usage });
            }
            Ok(None) => {
                // 蒸餾失敗，僅存原始
                if let Err(e) = db::postgres::insert_raw_message(&cfg.pg, &item.agent_id, &item.text, 0).await {
                    error!(agent_id = %item.agent_id, "寫入原始訊息失敗: {e:#}");
                }
            }
            Err(e) => {
                error!(agent_id = %item.agent_id, "蒸餾失敗: {e:#}");
                if let Err(e) = db::postgres::insert_raw_message(&cfg.pg, &item.agent_id, &item.text, 0).await {
                    error!(agent_id = %item.agent_id, "寫入原始訊息失敗: {e:#}");
                }
            }
        }
    }

    if !distilled.is_empty() {
        // 2. 批次 embedding
        let summaries: Vec<String> = distilled.iter().map(|d| d.data.dense_summary.clone()).collect();
        let vectors = match cfg.embedder.batch_embed(&summaries).await {
            Ok(v) => v,
            Err(e) => {
                error!("批次 embedding 失敗: {e:#}，逐筆 fallback");
                let mut vecs = Vec::with_capacity(summaries.len());
                for s in &summaries {
                    match cfg.embedder.embed(s).await {
                        Ok(v) => vecs.push(v),
                        Err(e) => {
                            error!("單筆 embedding 也失敗: {e:#}");
                            vecs.push(vec![]);
                        }
                    }
                }
                vecs
            }
        };

        // 3. 寫入 PG messages + Qdrant
        for (d, vector) in distilled.iter().zip(vectors) {
            if vector.is_empty() {
                if let Err(e) = db::postgres::insert_message(&cfg.pg, &d.row.agent_id, &d.data, 0).await {
                    error!(agent_id = %d.row.agent_id, "寫入 PG 失敗: {e:#}");
                }
                continue;
            }

            let inserted = match db::postgres::insert_message(&cfg.pg, &d.row.agent_id, &d.data, 0).await {
                Ok(ins) => ins,
                Err(e) => {
                    error!(agent_id = %d.row.agent_id, "寫入 PG 失敗: {e:#}");
                    continue;
                }
            };

            if let Err(e) = db::qdrant::upsert_point(
                &cfg.qdrant,
                inserted.id,
                vector,
                &d.row.agent_id,
                &d.data.original_text,
                &d.data.keywords,
                inserted.created_at.timestamp(),
            )
            .await
            {
                error!(agent_id = %d.row.agent_id, id = %inserted.id, "Qdrant upsert 失敗: {e:#}");
            }

            let _ = db::postgres::insert_token_usage(
                &cfg.pg,
                &d.row.agent_id,
                &d.row.worker_model,
                d.usage.input_tokens,
                d.usage.output_tokens,
            )
            .await;
        }
    }

    // 4. 刪除已處理的 pending 項目
    if let Err(e) = db::postgres::delete_pending_ingest(&cfg.pg, &pending_ids).await {
        error!("刪除 pending_ingest 失敗: {e:#}");
    }

    info!(count, "批次入庫完成");
}

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
