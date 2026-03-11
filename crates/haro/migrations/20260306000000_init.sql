-- Haro: LLM 記憶系統 PostgreSQL Schema
-- 前置條件：zhparser 擴充套件與 zhcn 搜尋配置已在容器初始化時建立
-- 見 deploy/init-extensions.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS messages (
    -- 主鍵，同時作為 Qdrant 的 point ID
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    -- 區分不同 Bot Agent 的記憶
    bot_id          VARCHAR(128) NOT NULL,
    -- LLM 蒸餾後的精煉摘要
    dense_summary   TEXT NOT NULL,
    -- LLM 提取的關鍵字
    keywords        TEXT[] NOT NULL DEFAULT '{}',
    -- 完整原文
    original_text   TEXT NOT NULL,
    -- 建立時間
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- 預計算 tsvector：摘要權重 A，原文權重 B
    -- 使用 zhparser 中文分詞配置
    search_vector   TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('zhcn', dense_summary), 'A') ||
        setweight(to_tsvector('zhcn', original_text), 'B')
    ) STORED
);

-- BM25 全文檢索 GIN 索引
CREATE INDEX IF NOT EXISTS idx_messages_search_vector
    ON messages USING GIN (search_vector);

-- bot_id 過濾索引
CREATE INDEX IF NOT EXISTS idx_messages_bot_id
    ON messages (bot_id);

-- 時間排序索引
CREATE INDEX IF NOT EXISTS idx_messages_created_at
    ON messages (created_at DESC);

-- 複合索引：bot_id + 時間排序
CREATE INDEX IF NOT EXISTS idx_messages_bot_id_created_at
    ON messages (bot_id, created_at DESC);
