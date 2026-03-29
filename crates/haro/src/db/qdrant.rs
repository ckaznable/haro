use std::collections::HashMap;

use anyhow::{Context, Result};
use chrono::DateTime;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType,
    Filter, ListValue, PointStruct, SearchPointsBuilder, UpsertPointsBuilder, Value,
    VectorParamsBuilder, point_id::PointIdOptions, value::Kind,
};
use tracing::info;
use uuid::Uuid;

use crate::models::SearchHit;

pub const COLLECTION_NAME: &str = "haro_memories";

// ── 初始化 ──

/// 建立 Qdrant 客戶端並初始化 Collection
pub async fn init_client(qdrant_url: &str, vector_dim: u64) -> Result<Qdrant> {
    let client = Qdrant::from_url(qdrant_url)
        .build()
        .context("無法建立 Qdrant 客戶端")?;

    info!("Qdrant 客戶端建立成功");

    init_collection(&client, vector_dim).await?;

    Ok(client)
}

async fn init_collection(client: &Qdrant, vector_dim: u64) -> Result<()> {
    let exists = client
        .collection_exists(COLLECTION_NAME)
        .await
        .context("檢查 Qdrant Collection 是否存在時失敗")?;

    if exists {
        info!("Qdrant Collection '{COLLECTION_NAME}' 已存在，跳過建立");
        return Ok(());
    }

    client
        .create_collection(
            CreateCollectionBuilder::new(COLLECTION_NAME).vectors_config(
                VectorParamsBuilder::new(vector_dim, Distance::Cosine).on_disk(true),
            ),
        )
        .await
        .context("建立 Qdrant Collection 失敗")?;

    info!("Qdrant Collection '{COLLECTION_NAME}' 建立完成 (dim={vector_dim}, mmap)");

    create_payload_indexes(client).await?;

    Ok(())
}

async fn create_payload_indexes(client: &Qdrant) -> Result<()> {
    client
        .create_field_index(CreateFieldIndexCollectionBuilder::new(
            COLLECTION_NAME,
            "bot_id",
            FieldType::Keyword,
        ))
        .await
        .context("建立 bot_id Payload 索引失敗")?;

    client
        .create_field_index(CreateFieldIndexCollectionBuilder::new(
            COLLECTION_NAME,
            "created_at",
            FieldType::Integer,
        ))
        .await
        .context("建立 created_at Payload 索引失敗")?;

    info!("Qdrant Payload 索引建立完成 (bot_id, created_at)");

    Ok(())
}

// ── 寫入 ──

/// 將向量與 Payload 寫入 Qdrant（point ID 與 PG 的 UUID 一致）
pub async fn upsert_point(
    client: &Qdrant,
    id: Uuid,
    vector: Vec<f32>,
    bot_id: &str,
    original_text: &str,
    keywords: &[String],
    created_at: i64,
) -> Result<()> {
    let kw_list = Value {
        kind: Some(Kind::ListValue(ListValue {
            values: keywords.iter().map(|k| Value::from(k.clone())).collect(),
        })),
    };

    let payload: HashMap<String, Value> = HashMap::from([
        ("bot_id".to_owned(), Value::from(bot_id.to_owned())),
        (
            "original_text".to_owned(),
            Value::from(original_text.to_owned()),
        ),
        ("keywords".to_owned(), kw_list),
        ("created_at".to_owned(), Value::from(created_at)),
    ]);

    client
        .upsert_points(UpsertPointsBuilder::new(
            COLLECTION_NAME,
            vec![PointStruct::new(id.to_string(), vector, payload)],
        ))
        .await
        .context("Qdrant upsert 失敗")?;

    Ok(())
}

// ── 向量搜尋 ──

/// 以向量進行語義搜尋，帶 bot_id 過濾
pub async fn search_vector(
    client: &Qdrant,
    bot_id: &str,
    query_vector: Vec<f32>,
    limit: u64,
) -> Result<Vec<SearchHit>> {
    let results = client
        .search_points(
            SearchPointsBuilder::new(COLLECTION_NAME, query_vector, limit)
                .filter(Filter::must([Condition::matches(
                    "bot_id",
                    bot_id.to_owned(),
                )]))
                .with_payload(true),
        )
        .await
        .context("Qdrant 向量搜尋失敗")?;

    let hits = results
        .result
        .into_iter()
        .filter_map(|point| {
            let uuid_str = match point.id?.point_id_options? {
                PointIdOptions::Uuid(s) => s,
                _ => return None,
            };
            let id = Uuid::parse_str(&uuid_str).ok()?;
            let original_text = extract_string(&point.payload, "original_text")?;
            let ts = extract_i64(&point.payload, "created_at")?;
            let created_at = DateTime::from_timestamp(ts, 0)?;

            Some(SearchHit {
                id,
                original_text,
                created_at,
            })
        })
        .collect();

    Ok(hits)
}

fn extract_string(payload: &HashMap<String, Value>, key: &str) -> Option<String> {
    match payload.get(key)?.kind.as_ref()? {
        Kind::StringValue(s) => Some(s.clone()),
        _ => None,
    }
}

fn extract_i64(payload: &HashMap<String, Value>, key: &str) -> Option<i64> {
    match payload.get(key)?.kind.as_ref()? {
        Kind::IntegerValue(i) => Some(*i),
        _ => None,
    }
}
