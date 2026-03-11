-- 儲存 Telegram 圖片的 file_id / 來源 chat_id / message_id，方便日後取回或轉發原圖
ALTER TABLE messages ADD COLUMN image_file_ids TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE messages ADD COLUMN source_chat_id BIGINT;
ALTER TABLE messages ADD COLUMN source_message_id INT;
