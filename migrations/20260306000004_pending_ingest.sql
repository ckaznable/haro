-- 批次入庫暫存表：訊息先寫入此表，背景任務處理後刪除
-- 若服務中斷，下次啟動可從此表恢復未處理的項目
CREATE TABLE IF NOT EXISTS pending_ingest (
    id         BIGSERIAL PRIMARY KEY,
    agent_id   TEXT        NOT NULL,
    text       TEXT        NOT NULL,
    worker_model TEXT      NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pending_ingest_created ON pending_ingest (created_at);
