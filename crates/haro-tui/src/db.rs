use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::postgres::PgPoolOptions;
use sqlx::{FromRow, PgPool};

pub async fn connect(url: &str) -> Result<PgPool> {
    PgPoolOptions::new()
        .max_connections(3)
        .connect(url)
        .await
        .context("無法連線至 PostgreSQL")
}

// ── Token Usage ──

#[derive(Debug, FromRow)]
pub struct TokenUsageRow {
    pub id: i64,
    pub agent_id: String,
    pub model: String,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub created_at: DateTime<Utc>,
}

/// 取得 token_usage 紀錄（時間倒序）
pub async fn get_token_usage(
    pool: &PgPool,
    agent_filter: Option<&str>,
    limit: i64,
    offset: i64,
) -> Result<Vec<TokenUsageRow>> {
    let rows = if let Some(agent) = agent_filter {
        sqlx::query_as::<_, TokenUsageRow>(
            "SELECT id, agent_id, model, input_tokens, output_tokens, created_at \
             FROM token_usage WHERE agent_id = $1 \
             ORDER BY created_at DESC LIMIT $2 OFFSET $3",
        )
        .bind(agent)
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query_as::<_, TokenUsageRow>(
            "SELECT id, agent_id, model, input_tokens, output_tokens, created_at \
             FROM token_usage \
             ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await?
    };
    Ok(rows)
}

// ── Daily Summary ──

#[derive(Debug, FromRow)]
pub struct DailySummary {
    pub day: DateTime<Utc>,
    pub agent_id: String,
    pub model: String,
    pub call_count: i64,
    pub total_input: i64,
    pub total_output: i64,
}

pub async fn get_daily_summary(
    pool: &PgPool,
    agent_filter: Option<&str>,
    days: i32,
) -> Result<Vec<DailySummary>> {
    let rows = if let Some(agent) = agent_filter {
        sqlx::query_as::<_, DailySummary>(
            "SELECT DATE_TRUNC('day', created_at) as day, agent_id, model, \
             COUNT(*)::bigint as call_count, \
             SUM(input_tokens)::bigint as total_input, \
             SUM(output_tokens)::bigint as total_output \
             FROM token_usage \
             WHERE agent_id = $1 AND created_at >= NOW() - make_interval(days => $2) \
             GROUP BY day, agent_id, model \
             ORDER BY day DESC, agent_id, model",
        )
        .bind(agent)
        .bind(days)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query_as::<_, DailySummary>(
            "SELECT DATE_TRUNC('day', created_at) as day, agent_id, model, \
             COUNT(*)::bigint as call_count, \
             SUM(input_tokens)::bigint as total_input, \
             SUM(output_tokens)::bigint as total_output \
             FROM token_usage \
             WHERE created_at >= NOW() - make_interval(days => $1) \
             GROUP BY day, agent_id, model \
             ORDER BY day DESC, agent_id, model",
        )
        .bind(days)
        .fetch_all(pool)
        .await?
    };
    Ok(rows)
}

// ── Agents 列表 ──

pub async fn get_agents(pool: &PgPool) -> Result<Vec<String>> {
    let rows: Vec<(String,)> =
        sqlx::query_as("SELECT DISTINCT agent_id FROM token_usage ORDER BY agent_id")
            .fetch_all(pool)
            .await?;
    Ok(rows.into_iter().map(|r| r.0).collect())
}

// ── Messages ──

#[derive(Debug, FromRow)]
pub struct MessageRow {
    pub id: uuid::Uuid,
    pub bot_id: String,
    pub dense_summary: String,
    pub original_text: String,
    pub token_count: i32,
    pub created_at: DateTime<Utc>,
}

pub async fn get_messages(
    pool: &PgPool,
    agent_filter: Option<&str>,
    limit: i64,
    offset: i64,
) -> Result<Vec<MessageRow>> {
    let rows = if let Some(agent) = agent_filter {
        sqlx::query_as::<_, MessageRow>(
            "SELECT id, bot_id, dense_summary, original_text, token_count, created_at \
             FROM messages WHERE bot_id = $1 \
             ORDER BY created_at DESC LIMIT $2 OFFSET $3",
        )
        .bind(agent)
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query_as::<_, MessageRow>(
            "SELECT id, bot_id, dense_summary, original_text, token_count, created_at \
             FROM messages \
             ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await?
    };
    Ok(rows)
}
