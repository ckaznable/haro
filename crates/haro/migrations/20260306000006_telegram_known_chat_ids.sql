CREATE TABLE IF NOT EXISTS telegram_known_chat_ids (
    channel_scope   VARCHAR(255) NOT NULL,
    chat_id         BIGINT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (channel_scope, chat_id)
);

CREATE INDEX IF NOT EXISTS idx_telegram_known_chat_ids_created_at
    ON telegram_known_chat_ids (created_at DESC);
