ALTER TABLE telegram_known_chat_ids
    RENAME TO notifier_targets;

ALTER TABLE notifier_targets
    RENAME COLUMN chat_id TO target_id;

ALTER TABLE notifier_targets
    ALTER COLUMN target_id TYPE TEXT USING target_id::TEXT;

ALTER TABLE notifier_targets
    ADD COLUMN IF NOT EXISTS display_name TEXT,
    ADD COLUMN IF NOT EXISTS metadata JSONB;

DROP INDEX IF EXISTS idx_telegram_known_chat_ids_created_at;
DROP INDEX IF EXISTS idx_telegram_known_chat_ids_last_seen_at;

CREATE INDEX IF NOT EXISTS idx_notifier_targets_created_at
    ON notifier_targets (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_notifier_targets_last_seen_at
    ON notifier_targets (last_seen_at DESC);
