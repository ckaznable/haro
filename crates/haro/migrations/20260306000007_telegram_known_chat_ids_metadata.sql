ALTER TABLE telegram_known_chat_ids
    ADD COLUMN IF NOT EXISTS username TEXT,
    ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

UPDATE telegram_known_chat_ids
SET last_seen_at = COALESCE(last_seen_at, created_at)
WHERE last_seen_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_telegram_known_chat_ids_last_seen_at
    ON telegram_known_chat_ids (last_seen_at DESC);
