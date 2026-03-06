use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// LLM 資料蒸餾輸出
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledData {
    pub dense_summary: String,
    pub keywords: Vec<String>,
    pub original_text: String,
}

/// 搜尋命中（PG BM25 / Qdrant 向量搜尋共用）
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: Uuid,
    pub original_text: String,
    pub created_at: DateTime<Utc>,
}

/// RRF 融合排序後的最終結果
#[derive(Debug, Clone)]
pub struct RankedResult {
    pub id: Uuid,
    pub score: f64,
    pub original_text: String,
}
