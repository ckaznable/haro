-- 為 messages 表新增 token_count 欄位，記錄原始文本的 token 數量
ALTER TABLE messages ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0;
