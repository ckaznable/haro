use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::postgres::PgPoolOptions;
use sqlx::{FromRow, PgPool};
use tracing::info;
use uuid::Uuid;

use crate::models::{DistilledData, SearchHit};

/// 建立 PostgreSQL 連線池並執行資料庫遷移
pub async fn init_pool(database_url: &str) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await
        .context("無法連線至 PostgreSQL")?;

    info!("PostgreSQL 連線成功");

    sqlx::migrate!()
        .run(&pool)
        .await
        .context("資料庫遷移失敗")?;

    info!("資料庫遷移完成");

    Ok(pool)
}

// ── 寫入 ──

#[derive(FromRow)]
struct InsertedRow {
    id: Uuid,
    created_at: DateTime<Utc>,
}

pub struct InsertedMessage {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
}

/// 圖片來源 metadata
pub struct ImageMeta<'a> {
    pub file_ids: &'a [String],
    pub source_chat_id: Option<i64>,
    pub source_message_id: Option<i32>,
}

/// 將蒸餾後的資料寫入 messages 表
pub async fn insert_message(
    pool: &PgPool,
    bot_id: &str,
    data: &DistilledData,
    token_count: i32,
    image_meta: &ImageMeta<'_>,
) -> Result<InsertedMessage> {
    let row = sqlx::query_as::<_, InsertedRow>(
        "INSERT INTO messages (bot_id, dense_summary, keywords, original_text, token_count, image_file_ids, source_chat_id, source_message_id) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8) \
         RETURNING id, created_at",
    )
    .bind(bot_id)
    .bind(&data.dense_summary)
    .bind(&data.keywords)
    .bind(&data.original_text)
    .bind(token_count)
    .bind(image_meta.file_ids)
    .bind(image_meta.source_chat_id)
    .bind(image_meta.source_message_id)
    .fetch_one(pool)
    .await
    .context("插入訊息失敗")?;

    Ok(InsertedMessage {
        id: row.id,
        created_at: row.created_at,
    })
}

/// 寫入原始訊息（不經蒸餾，僅保留歷史）
pub async fn insert_raw_message(
    pool: &PgPool,
    bot_id: &str,
    text: &str,
    token_count: i32,
) -> Result<InsertedMessage> {
    let row = sqlx::query_as::<_, InsertedRow>(
        "INSERT INTO messages (bot_id, dense_summary, keywords, original_text, token_count) \
         VALUES ($1, '', '{}', $2, $3) \
         RETURNING id, created_at",
    )
    .bind(bot_id)
    .bind(text)
    .bind(token_count)
    .fetch_one(pool)
    .await
    .context("插入原始訊息失敗")?;

    Ok(InsertedMessage {
        id: row.id,
        created_at: row.created_at,
    })
}

// ── BM25 搜尋 ──

#[derive(FromRow)]
struct BM25Row {
    id: Uuid,
    original_text: String,
    created_at: DateTime<Utc>,
}

/// 在 messages 表進行 BM25 全文檢索（zhcn 分詞）
pub async fn search_bm25(
    pool: &PgPool,
    bot_id: &str,
    query: &str,
    limit: i64,
) -> Result<Vec<SearchHit>> {
    let rows = sqlx::query_as::<_, BM25Row>(
        "SELECT m.id, m.original_text, m.created_at \
         FROM messages m, plainto_tsquery('zhcn', $2) q \
         WHERE m.bot_id = $1 AND m.search_vector @@ q \
         ORDER BY ts_rank(m.search_vector, q) DESC \
         LIMIT $3",
    )
    .bind(bot_id)
    .bind(query)
    .bind(limit)
    .fetch_all(pool)
    .await
    .context("BM25 搜尋失敗")?;

    Ok(rows
        .into_iter()
        .map(|r| SearchHit {
            id: r.id,
            original_text: r.original_text,
            created_at: r.created_at,
        })
        .collect())
}

// ── 對話歷史 ──

/// 取得近期訊息作為對話歷史，累積 token 數不超過 max_tokens
/// 回傳按時間正序（舊→新），每筆為 original_text
pub async fn get_history(pool: &PgPool, bot_id: &str, max_tokens: i32) -> Result<Vec<String>> {
    // 用視窗函式累積 token，在 SQL 層就截斷
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT original_text FROM ( \
             SELECT original_text, created_at, \
                    SUM(token_count) OVER (ORDER BY created_at DESC) AS cumulative_tokens \
             FROM messages \
             WHERE bot_id = $1 AND token_count > 0 \
             ORDER BY created_at DESC \
         ) sub \
         WHERE cumulative_tokens <= $2 \
         ORDER BY created_at ASC",
    )
    .bind(bot_id)
    .bind(max_tokens)
    .fetch_all(pool)
    .await
    .context("讀取對話歷史失敗")?;

    Ok(rows.into_iter().map(|(t,)| t).collect())
}

// ── Token 用量 ──

/// 記錄一次 LLM 呼叫的 token 用量
pub async fn insert_token_usage(
    pool: &PgPool,
    agent_id: &str,
    model: &str,
    input_tokens: i32,
    output_tokens: i32,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO token_usage (agent_id, model, input_tokens, output_tokens) \
         VALUES ($1, $2, $3, $4)",
    )
    .bind(agent_id)
    .bind(model)
    .bind(input_tokens)
    .bind(output_tokens)
    .execute(pool)
    .await
    .context("寫入 token 用量失敗")?;

    Ok(())
}

// ── 批次入庫暫存 ──

/// 將訊息寫入 pending_ingest 暫存表
pub async fn insert_pending_ingest(
    pool: &PgPool,
    agent_id: &str,
    text: &str,
    worker_model: &str,
) -> Result<()> {
    sqlx::query("INSERT INTO pending_ingest (agent_id, text, worker_model) VALUES ($1, $2, $3)")
        .bind(agent_id)
        .bind(text)
        .bind(worker_model)
        .execute(pool)
        .await
        .context("寫入 pending_ingest 失敗")?;

    Ok(())
}

/// 暫存項目
#[derive(FromRow)]
pub struct PendingIngestRow {
    pub id: i64,
    pub agent_id: String,
    pub text: String,
    pub worker_model: String,
}

/// 取出最多 limit 筆待處理項目（按建立時間排序）
pub async fn fetch_pending_ingest(pool: &PgPool, limit: i64) -> Result<Vec<PendingIngestRow>> {
    let rows = sqlx::query_as::<_, PendingIngestRow>(
        "SELECT id, agent_id, text, worker_model FROM pending_ingest ORDER BY created_at LIMIT $1",
    )
    .bind(limit)
    .fetch_all(pool)
    .await
    .context("讀取 pending_ingest 失敗")?;

    Ok(rows)
}

/// 取得某 agent 的所有待處理文字（供查詢時補充上下文）
pub async fn get_pending_texts(pool: &PgPool, agent_id: &str) -> Result<Vec<String>> {
    let rows: Vec<(String,)> =
        sqlx::query_as("SELECT text FROM pending_ingest WHERE agent_id = $1 ORDER BY created_at")
            .bind(agent_id)
            .fetch_all(pool)
            .await
            .context("讀取 pending_ingest 文字失敗")?;

    Ok(rows.into_iter().map(|r| r.0).collect())
}

/// 刪除已處理的暫存項目
pub async fn delete_pending_ingest(pool: &PgPool, ids: &[i64]) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    sqlx::query("DELETE FROM pending_ingest WHERE id = ANY($1)")
        .bind(ids)
        .execute(pool)
        .await
        .context("刪除 pending_ingest 失敗")?;

    Ok(())
}

// ── 備忘錄 ──

/// 寫入或更新備忘錄（upsert）
pub async fn upsert_scratchpad(pool: &PgPool, bot_id: &str, content: &str) -> Result<()> {
    sqlx::query(
        "INSERT INTO scratchpad (bot_id, content, updated_at) \
         VALUES ($1, $2, NOW()) \
         ON CONFLICT (bot_id) DO UPDATE SET content = $2, updated_at = NOW()",
    )
    .bind(bot_id)
    .bind(content)
    .execute(pool)
    .await
    .context("寫入備忘錄失敗")?;

    Ok(())
}

/// 讀取備忘錄
pub async fn get_scratchpad(pool: &PgPool, bot_id: &str) -> Result<Option<String>> {
    let row: Option<(String,)> = sqlx::query_as("SELECT content FROM scratchpad WHERE bot_id = $1")
        .bind(bot_id)
        .fetch_optional(pool)
        .await
        .context("讀取備忘錄失敗")?;

    Ok(row.map(|r| r.0))
}

// ── Notifier targets ──

/// 讀取指定 channel scope 已持久化的 notifier target IDs
pub async fn list_notifier_target_ids(pool: &PgPool, channel_scope: &str) -> Result<Vec<String>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT target_id \
         FROM notifier_targets \
         WHERE channel_scope = $1 \
         ORDER BY created_at ASC",
    )
    .bind(channel_scope)
    .fetch_all(pool)
    .await
    .context("讀取 notifier targets 失敗")?;

    Ok(rows.into_iter().map(|(target_id,)| target_id).collect())
}

/// 寫入 notifier target；重複寫入時更新可選 metadata 與 last_seen_at
pub async fn upsert_notifier_target(
    pool: &PgPool,
    channel_scope: &str,
    target_id: &str,
    username: Option<&str>,
    display_name: Option<&str>,
    metadata: Option<serde_json::Value>,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO notifier_targets \
            (channel_scope, target_id, username, display_name, metadata, last_seen_at) \
         VALUES ($1, $2, $3, $4, $5, NOW()) \
         ON CONFLICT (channel_scope, target_id) DO UPDATE \
         SET username = COALESCE(EXCLUDED.username, notifier_targets.username), \
             display_name = COALESCE(EXCLUDED.display_name, notifier_targets.display_name), \
             metadata = COALESCE(EXCLUDED.metadata, notifier_targets.metadata), \
             last_seen_at = NOW()",
    )
    .bind(channel_scope)
    .bind(target_id)
    .bind(username)
    .bind(display_name)
    .bind(metadata)
    .execute(pool)
    .await
    .context("寫入 notifier target 失敗")?;

    Ok(())
}
