mod cmd;
mod heartbeat;
mod scheduler;

use std::sync::Arc;

use anyhow::Result;
use tokio::task::JoinHandle;
use tracing::info;

use crate::agent::{Agent, AgentMode};
use crate::api::{self};
use crate::channel::{MessageHandler, Notifier};
use crate::config::AppConfig;
use crate::db::postgres::ImageMeta;
use crate::{db, search, tool};

/// 共用資源，傳遞給各 agent 的 channel 和心跳任務
pub struct SharedResources {
    pub pg: Arc<sqlx::PgPool>,
    pub qdrant: Arc<qdrant_client::Qdrant>,
    pub embedder: Arc<api::gemini::GeminiProvider>,
    pub llm: Arc<api::gemini::GeminiProvider>,
    pub worker: Arc<api::gemini::GeminiProvider>,
    pub brain: Arc<api::gemini::GeminiProvider>,
}

/// 啟動所有 agent 的 channel、心跳等任務，回傳 JoinHandle 列表
pub fn spawn_all(
    agents: Vec<Agent>,
    res: &SharedResources,
    cfg: &AppConfig,
) -> Vec<JoinHandle<Result<()>>> {
    let mut handles = Vec::new();

    let heartbeat_interval = cfg.heartbeat_interval;
    let brain_model = cfg
        .brain
        .as_ref()
        .map(|b| b.model.clone())
        .unwrap_or_else(|| cfg.llm.model.clone());

    for agent in agents {
        let agent_id = agent.id;
        let agent_mode = agent.mode;
        let agent_prompt = agent.prompt;
        let agent_soul = agent.soul;
        let agent_heartbeat = agent.heartbeat;
        let agent_brain_heartbeat = agent.brain_heartbeat;
        let agent_path = agent.path;
        let agent_llm_commands = agent.llm_commands;

        // 收集所有頻道的 notifier（在 channel.start() 消耗 channel 前）
        let notifiers: Vec<Arc<dyn Notifier>> = agent
            .channels
            .iter()
            .filter_map(|ch| ch.notifier())
            .map(Arc::from)
            .collect();

        // 啟動 LLM 心跳任務
        if !agent_heartbeat.is_empty() {
            info!(agent_id = %agent_id, interval = heartbeat_interval, "啟動 LLM 心跳任務");
            handles.push(tokio::spawn(heartbeat::run_heartbeat(
                heartbeat::HeartbeatTask {
                    pg: Arc::clone(&res.pg),
                    worker: Arc::clone(&res.worker),
                    llm: Arc::clone(&res.llm),
                    agent_id: agent_id.clone(),
                    agent_prompt: agent_prompt.clone(),
                    heartbeat_prompt: agent_heartbeat,
                    interval_secs: heartbeat_interval,
                    llm_model: cfg.llm.model.clone(),
                    notifiers: notifiers.clone(),
                    searxng_url: cfg.searxng_url.clone(),
                    saachi_url: cfg.saachi_url.clone(),
                    agent_path: agent_path.clone(),
                },
            )));
        }

        // 啟動大腦心跳任務
        if !agent_brain_heartbeat.is_empty() {
            info!(agent_id = %agent_id, interval = heartbeat_interval, "啟動大腦心跳任務");
            handles.push(tokio::spawn(heartbeat::run_heartbeat(
                heartbeat::HeartbeatTask {
                    pg: Arc::clone(&res.pg),
                    worker: Arc::clone(&res.worker),
                    llm: Arc::clone(&res.brain),
                    agent_id: agent_id.clone(),
                    agent_prompt: agent_prompt.clone(),
                    heartbeat_prompt: agent_brain_heartbeat,
                    interval_secs: heartbeat_interval,
                    llm_model: brain_model.clone(),
                    notifiers: notifiers.clone(),
                    searxng_url: cfg.searxng_url.clone(),
                    saachi_url: cfg.saachi_url.clone(),
                    agent_path: agent_path.clone(),
                },
            )));
        }

        // Skills 目錄（存在才啟用）
        let skills_path = agent_path
            .as_ref()
            .map(|p| p.join("skills"))
            .filter(|p| p.is_dir());

        // 建立指令註冊表
        let cmd_registry = cmd::register_all(&cmd::CmdContext {
            res,
            cfg,
            agent_id: &agent_id,
            agent_mode: &agent_mode,
            agent_prompt: &agent_prompt,
            agent_soul: &agent_soul,
            agent_path: &agent_path,
            skills_path: &skills_path,
            notifiers: &notifiers,
            llm_commands: &agent_llm_commands,
            brain_model: &brain_model,
        });

        let commands = Arc::new(cmd_registry);

        // ingest 模式：建立批次入庫佇列 + 背景任務
        let ingest_queue = if agent_mode == AgentMode::Ingest {
            let (queue, batch_handle) = search::spawn_batch_ingest(search::BatchIngestConfig {
                pg: Arc::clone(&res.pg),
                qdrant: Arc::clone(&res.qdrant),
                embedder: Arc::clone(&res.embedder),
                worker: Arc::clone(&res.worker),
                batch_size: 10,
                flush_interval: std::time::Duration::from_secs(5),
            });
            handles.push(batch_handle);
            info!(agent_id = %agent_id, "批次入庫背景任務已啟動");
            Some(queue)
        } else {
            None
        };

        for channel in agent.channels {
            let msg_ctx = Arc::new(MessageContext {
                pg: Arc::clone(&res.pg),
                qdrant: Arc::clone(&res.qdrant),
                embedder: Arc::clone(&res.embedder),
                llm: Arc::clone(&res.llm),
                worker: Arc::clone(&res.worker),
                agent_id: agent_id.clone(),
                prompt: agent_prompt.clone(),
                soul: agent_soul.clone(),
                worker_model: cfg.worker.model.clone(),
                llm_model: cfg.llm.model.clone(),
                image_embed: cfg.embedding.image_embed,
                skills_path: skills_path.clone(),
                agent_path: agent_path.clone(),
                searxng_url: cfg.searxng_url.clone(),
                saachi_url: cfg.saachi_url.clone(),
            });

            let mode = agent_mode.clone();
            let queue = ingest_queue.clone();
            let ch_notifiers = notifiers.clone();
            let handler: MessageHandler = Arc::new(move |msg| {
                let ctx = Arc::clone(&msg_ctx);
                let mode = mode.clone();
                let queue = queue.clone();
                let notifiers = ch_notifiers.clone();
                Box::pin(async move {
                    let result = match mode {
                        AgentMode::Chat => handle_message(&ctx, &msg).await,
                        AgentMode::Ingest => {
                            if let Some(q) = &queue {
                                q.push(&ctx.agent_id, &msg.text, &ctx.worker_model).await?;
                            } else {
                                let images = to_image_inputs(&msg.images);
                                let (file_ids, source_chat_id, source_message_id) =
                                    extract_image_meta(&msg.images);
                                let image_meta = ImageMeta {
                                    file_ids: &file_ids,
                                    source_chat_id,
                                    source_message_id,
                                };
                                handle_ingest(&ctx, &msg.text, &images, &image_meta).await?;
                            }
                            Ok(None)
                        }
                    };

                    // 偵測 API 額度耗盡，廣播通知所有頻道
                    if let Err(ref e) = result {
                        let msg = format!("{e:#}");
                        if msg.contains("429") && msg.contains("RESOURCE_EXHAUSTED") {
                            let notice = "⚠️ API 額度已用盡，服務暫時無法使用。";
                            for n in &notifiers {
                                let _ = n.send(notice).await;
                            }
                        }
                    }

                    result
                })
            });

            let cmds = Arc::clone(&commands);
            handles.push(tokio::spawn(channel.start(handler, cmds)));
        }
    }

    handles
}

// ── 訊息處理 ──

/// 訊息處理所需的共用上下文（避免過多參數）
struct MessageContext {
    pg: Arc<sqlx::PgPool>,
    qdrant: Arc<qdrant_client::Qdrant>,
    embedder: Arc<api::gemini::GeminiProvider>,
    llm: Arc<api::gemini::GeminiProvider>,
    worker: Arc<api::gemini::GeminiProvider>,
    agent_id: String,
    prompt: String,
    soul: String,
    worker_model: String,
    llm_model: String,
    image_embed: bool,
    /// Agent 的 skills 目錄路徑（None = 無 skills）
    skills_path: Option<std::path::PathBuf>,
    /// Agent 的根目錄路徑（None = 無 heartbeat 工具）
    agent_path: Option<std::path::PathBuf>,
    /// SearXNG base URL（None = 不啟用）
    searxng_url: Option<String>,
    /// Saachi base URL（None = 不啟用）
    saachi_url: Option<String>,
}

/// 將 channel::ImageData 轉換為 api::ImageInput
fn to_image_inputs(images: &[crate::channel::ImageData]) -> Vec<api::ImageInput> {
    images
        .iter()
        .map(|img| api::ImageInput {
            mime_type: img.mime_type.clone(),
            data: img.data.clone(),
        })
        .collect()
}

/// 從 channel::ImageData 提取圖片來源 metadata
fn extract_image_meta(
    images: &[crate::channel::ImageData],
) -> (Vec<String>, Option<i64>, Option<i32>) {
    let file_ids: Vec<String> = images
        .iter()
        .filter_map(|img| img.file_id.clone())
        .collect();
    // 取第一張圖片的來源資訊（同一則訊息的圖片來自同一個 chat/message）
    let source_chat_id = images.first().and_then(|img| img.source_chat_id);
    let source_message_id = images.first().and_then(|img| img.source_message_id);
    (file_ids, source_chat_id, source_message_id)
}

/// 僅入庫（ingest 模式），不呼叫 LLM 回覆
async fn handle_ingest(
    ctx: &MessageContext,
    text: &str,
    images: &[api::ImageInput],
    image_meta: &ImageMeta<'_>,
) -> Result<()> {
    let embed_images = if ctx.image_embed { images } else { &[] };
    let (_, wk_usage) = search::ingest(
        &ctx.pg,
        &ctx.qdrant,
        ctx.embedder.as_ref(),
        ctx.worker.as_ref(),
        &ctx.agent_id,
        text,
        embed_images,
        image_meta,
    )
    .await?;

    db::postgres::insert_token_usage(
        &ctx.pg,
        &ctx.agent_id,
        &ctx.worker_model,
        wk_usage.input_tokens,
        wk_usage.output_tokens,
    )
    .await?;

    Ok(())
}

/// 查詢（不入庫）：檢索記憶 + LLM 回答
async fn handle_query(
    ctx: &MessageContext,
    question: &str,
    images: &[api::ImageInput],
    progress: Option<&api::ProgressSender>,
) -> Result<String> {
    let memo = db::postgres::get_scratchpad(&ctx.pg, &ctx.agent_id)
        .await?
        .unwrap_or_default();

    let results = search::retrieve(
        &ctx.pg,
        &ctx.qdrant,
        ctx.embedder.as_ref(),
        &ctx.agent_id,
        question,
        5,
    )
    .await?;

    let mut tools = tool::ToolRegistry::new();
    for t in tool::scratchpad::tools(Arc::clone(&ctx.pg), ctx.agent_id.clone()) {
        tools.register(t);
    }
    tools.register(tool::fetch::tool());
    if ctx.llm.grounding() {
        tools.register(tool::search::tool(Arc::clone(&ctx.llm)));
    }
    if let Some(ref url) = ctx.searxng_url {
        tools.register(tool::searxng::tool(url.clone()));
    }
    if let Some(ref url) = ctx.saachi_url {
        tools.register(tool::saachi::tool(url.clone()));
    }

    // 註冊 heartbeat + cron 工具（如有 agent 目錄）
    if let Some(ref ap) = ctx.agent_path {
        for t in tool::heartbeat::tools(ap.clone()) {
            tools.register(t);
        }
        for t in tool::cron::tools(ap.clone()) {
            tools.register(t);
        }
        for t in tool::task::tools(ap.clone()) {
            tools.register(t);
        }
    }

    // 註冊 skills 工具（如有 skills 目錄）
    let has_skills = ctx.skills_path.is_some();
    if let Some(ref sp) = ctx.skills_path {
        for t in tool::skills::tools(sp.clone()) {
            tools.register(t);
        }
    }

    let context_str: String = results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("[記憶 {}] {}", i + 1, r.original_text))
        .collect::<Vec<_>>()
        .join("\n\n");

    // 近期對話歷史（token-based，最多 5000 tokens）
    let history = db::postgres::get_history(&ctx.pg, &ctx.agent_id, 5000)
        .await
        .unwrap_or_default();
    let history_section = if history.is_empty() {
        String::new()
    } else {
        let items: String = history.join("\n\n");
        format!("\n\n## 近期對話\n{items}")
    };

    // 尚未蒸餾的 pending 訊息也加入上下文
    let pending_texts = db::postgres::get_pending_texts(&ctx.pg, &ctx.agent_id)
        .await
        .unwrap_or_default();
    let pending_section = if pending_texts.is_empty() {
        String::new()
    } else {
        let items: String = pending_texts
            .iter()
            .enumerate()
            .map(|(i, t)| format!("[待處理 {}] {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n\n");
        format!("\n\n## 待處理資料（尚未索引）\n{items}")
    };

    let memo_section = if memo.is_empty() {
        "（空）".to_owned()
    } else {
        memo
    };
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %:z");
    let soul_section = if ctx.soul.is_empty() {
        String::new()
    } else {
        format!("## 性格設定\n{}\n\n", ctx.soul)
    };

    let skills_instruction = if has_skills {
        "\n如果使用者要求使用某個 skill，請先使用 list_skills 工具查看可用的 skills，\
         再使用 get_skill 工具載入該 skill 的內容並依照其指示執行。"
    } else {
        ""
    };

    let system = format!(
        "你是一個知識庫助手，根據已收集的資料回答問題。\n\n\
         目前時間：{now}\n\n\
         {soul_section}\
         {}\n\n\
         ## 備忘錄 (Scratchpad)\n{memo_section}\n\n\
         ## 相關資料\n{context_str}{pending_section}{history_section}\n\n\
         ## 指示\n\
         請根據上述資料回答使用者的問題。\n\
         如果資料中沒有相關內容，請如實告知。\n\
         如果有重要的事情需要記住，請使用 save_memo 工具儲存。\n\
         備忘錄只用於紀錄固定且明確的資訊（例如偏好、規則、待辦事項），不要紀錄曾經做過的事情。{skills_instruction}",
        ctx.prompt
    );

    let result = api::chat_with_images(
        ctx.llm.as_ref(),
        &system,
        question,
        images,
        &tools,
        progress,
    )
    .await?;

    db::postgres::insert_token_usage(
        &ctx.pg,
        &ctx.agent_id,
        &ctx.llm_model,
        result.input_tokens,
        result.output_tokens,
    )
    .await?;

    Ok(result.text)
}

/// 處理單則訊息（chat 模式：先回覆，再入庫使用者訊息與 LLM 回覆）
async fn handle_message(
    ctx: &MessageContext,
    msg: &crate::channel::IncomingMessage,
) -> Result<Option<String>> {
    let images = to_image_inputs(&msg.images);
    let (file_ids, source_chat_id, source_message_id) = extract_image_meta(&msg.images);
    let image_meta = ImageMeta {
        file_ids: &file_ids,
        source_chat_id,
        source_message_id,
    };

    // 1. 先查詢 + 回覆（LLM chat 端一律傳圖，不受 image_embed 影響）
    let reply = handle_query(ctx, &msg.text, &images, msg.progress.as_ref()).await?;

    // 2. 入庫使用者訊息（image_embed 由 handle_ingest 內部判斷）
    let empty_meta = ImageMeta {
        file_ids: &[],
        source_chat_id: None,
        source_message_id: None,
    };
    handle_ingest(ctx, &msg.text, &images, &image_meta).await?;

    // 3. 入庫 LLM 回覆
    handle_ingest(ctx, &reply, &[], &empty_meta).await?;

    Ok(Some(reply))
}
