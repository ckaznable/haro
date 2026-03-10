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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distilled_data_deserialize() {
        let json = r#"{
            "dense_summary": "Rust async runtime",
            "keywords": ["rust", "async", "tokio"],
            "original_text": "Tokio is an async runtime for Rust."
        }"#;

        let data: DistilledData = serde_json::from_str(json).unwrap();
        assert_eq!(data.dense_summary, "Rust async runtime");
        assert_eq!(data.keywords, vec!["rust", "async", "tokio"]);
        assert_eq!(data.original_text, "Tokio is an async runtime for Rust.");
    }

    #[test]
    fn distilled_data_roundtrip() {
        let data = DistilledData {
            dense_summary: "summary".into(),
            keywords: vec!["a".into(), "b".into()],
            original_text: "original".into(),
        };
        let json = serde_json::to_string(&data).unwrap();
        let back: DistilledData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dense_summary, data.dense_summary);
        assert_eq!(back.keywords, data.keywords);
    }

    #[test]
    fn distilled_data_empty_keywords() {
        let json = r#"{"dense_summary": "s", "keywords": [], "original_text": "o"}"#;
        let data: DistilledData = serde_json::from_str(json).unwrap();
        assert!(data.keywords.is_empty());
    }

    #[test]
    fn distilled_data_missing_field_fails() {
        let json = r#"{"dense_summary": "s"}"#;
        let result: Result<DistilledData, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}
