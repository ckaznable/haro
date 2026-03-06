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

/// 將蒸餾後的資料寫入 messages 表
pub async fn insert_message(
    pool: &PgPool,
    bot_id: &str,
    data: &DistilledData,
    token_count: i32,
) -> Result<InsertedMessage> {
    let row = sqlx::query_as::<_, InsertedRow>(
        "INSERT INTO messages (bot_id, dense_summary, keywords, original_text, token_count) \
         VALUES ($1, $2, $3, $4, $5) \
         RETURNING id, created_at",
    )
    .bind(bot_id)
    .bind(&data.dense_summary)
    .bind(&data.keywords)
    .bind(&data.original_text)
    .bind(token_count)
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

// ── 近期歷史 ──

/// 取得最近 N 筆原始訊息及其 token 數量（時間倒序）
pub async fn get_recent_messages(
    pool: &PgPool,
    bot_id: &str,
    limit: i64,
) -> Result<Vec<(String, i32)>> {
    let rows: Vec<(String, i32)> = sqlx::query_as(
        "SELECT original_text, token_count FROM messages \
         WHERE bot_id = $1 \
         ORDER BY created_at DESC \
         LIMIT $2",
    )
    .bind(bot_id)
    .bind(limit)
    .fetch_all(pool)
    .await
    .context("讀取近期訊息失敗")?;

    Ok(rows)
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
    let row: Option<(String,)> =
        sqlx::query_as("SELECT content FROM scratchpad WHERE bot_id = $1")
            .bind(bot_id)
            .fetch_optional(pool)
            .await
            .context("讀取備忘錄失敗")?;

    Ok(row.map(|r| r.0))
}
