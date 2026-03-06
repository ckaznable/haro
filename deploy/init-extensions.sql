-- 容器首次啟動時自動執行（/docker-entrypoint-initdb.d/）

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS zhparser;

-- 建立中文全文搜尋配置
CREATE TEXT SEARCH CONFIGURATION zhcn (PARSER = zhparser);
ALTER TEXT SEARCH CONFIGURATION zhcn ADD MAPPING FOR
    n,v,a,i,e,l,j WITH simple;
