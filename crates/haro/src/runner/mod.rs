mod heartbeat;
mod scheduler;

use std::sync::Arc;

use anyhow::Result;
use tokio::task::JoinHandle;
use tracing::info;

use crate::agent::{Agent, AgentMode};
use crate::api::{self, LlmProvider};
use crate::channel::{CommandRegistry, MessageHandler, Notifier};
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
            handles.push(tokio::spawn(heartbeat::run_heartbeat(heartbeat::HeartbeatTask {
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
                agent_path: agent_path.clone(),
            })));
        }

        // 啟動大腦心跳任務
        if !agent_brain_heartbeat.is_empty() {
            info!(agent_id = %agent_id, interval = heartbeat_interval, "啟動大腦心跳任務");
            handles.push(tokio::spawn(heartbeat::run_heartbeat(heartbeat::HeartbeatTask {
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
                agent_path: agent_path.clone(),
            })));
        }

        // 建立指令註冊表
        let mut cmd_registry = CommandRegistry::new();

        cmd_registry.register(
            "stop",
            "中斷正在處理的任務",
            "/stop",
            Arc::new(|_sender, _args| Box::pin(async { Ok("此指令由頻道直接處理。".into()) })),
        );

        cmd_registry.register(
            "ping",
            "檢查機器人是否在線",
            "/ping",
            Arc::new(|_sender, _args| Box::pin(async { Ok("OK".into()) })),
        );

        // /memo：查看或編輯備忘錄
        {
            let memo_pg = Arc::clone(&res.pg);
            let memo_agent_id = agent_id.clone();
            cmd_registry.register(
                "memo",
                "查看或編輯備忘錄",
                "/memo [set <內容>]",
                Arc::new(move |_sender, args| {
                    let pg = Arc::clone(&memo_pg);
                    let aid = memo_agent_id.clone();
                    Box::pin(async move {
                        if let Some(new_content) = args.strip_prefix("set ") {
                            db::postgres::upsert_scratchpad(&pg, &aid, new_content.trim()).await?;
                            Ok("📝 備忘錄已更新".into())
                        } else if args.trim().is_empty() || args.trim() == "show" {
                            let content = db::postgres::get_scratchpad(&pg, &aid)
                                .await?
                                .unwrap_or_else(|| "（空）".into());
                            Ok(format!("📝 備忘錄：\n\n{content}"))
                        } else {
                            Ok("用法: /memo [show] | /memo set <內容>".into())
                        }
                    })
                }),
            );
        }

        // 心跳查看/編輯指令
        if let Some(ref apath) = agent_path {
            register_heartbeat_commands(&mut cmd_registry, apath);
        }

        // Skills 目錄（存在才啟用）
        let skills_path = agent_path
            .as_ref()
            .map(|p| p.join("skills"))
            .filter(|p| p.is_dir());

        // /cron 指令 + 排程器
        if let Some(ref apath) = agent_path {
            let cron_path = apath.join("cron.toml");
            let tasks_path = apath.join("tasks.toml");

            // 建立排程執行池
            let pool = scheduler::RunningPool::new(scheduler::SchedulerResources {
                pg: Arc::clone(&res.pg),
                worker: Arc::clone(&res.worker),
                llm: Arc::clone(&res.llm),
                brain: Arc::clone(&res.brain),
                agent_id: agent_id.clone(),
                agent_prompt: agent_prompt.clone(),
                tasks_path: tasks_path.clone(),
                llm_model: cfg.llm.model.clone(),
                brain_model: brain_model.clone(),
                worker_model: cfg.worker.model.clone(),
                notifiers: notifiers.clone(),
                searxng_url: cfg.searxng_url.clone(),
                skills_path: skills_path.clone(),
            });

            // 載入現有排程並 spawn
            info!(agent_id = %agent_id, "啟動排程任務");
            pool.spawn_all(&cron_path, &tasks_path);

            // /cron 指令
            let cp = cron_path.clone();
            let cron_worker = Arc::clone(&res.worker);
            let cron_pool = Arc::clone(&pool);
            cmd_registry.register(
                "cron",
                "管理排程任務（支援自然語言）",
                "/cron [list|add|remove|enable|disable|<自然語言>]",
                Arc::new(move |_sender, args| {
                    let cp = cp.clone();
                    let worker = Arc::clone(&cron_worker);
                    let pool = Arc::clone(&cron_pool);
                    Box::pin(async move {
                        // 先嘗試結構化指令
                        let result = tool::cron::handle_slash_command(&cp, &args);
                        match result {
                            // 回傳「用法」表示格式不對，改用 worker 解讀
                            Ok(ref text) if text.starts_with("用法:") => {}
                            other => {
                                let trimmed = args.trim();
                                // 新增時同步 spawn
                                if trimmed.starts_with("add ")
                                    && let Some(job) = tool::cron::load_config(&cp)
                                        .ok()
                                        .and_then(|c| c.jobs.last().cloned())
                                {
                                    pool.spawn_cron(&job);
                                }
                                // 刪除時中止
                                if (trimmed.starts_with("remove ") || trimmed.starts_with("rm "))
                                    && let Ok(text) = &other
                                    && let Some(id) = extract_removed_id(text, "排程任務")
                                {
                                    pool.abort(&id);
                                }
                                return other;
                            }
                        }
                        // fallback: 用 worker + cron 工具解讀自然語言
                        let mut tools = tool::ToolRegistry::new();
                        for t in tool::cron::tools(cp.parent().unwrap().to_path_buf()) {
                            tools.register(t);
                        }
                        let system = "你是一個排程任務管理助手。\
                            使用者想管理 cron 排程任務，請根據使用者的描述使用工具完成操作。\
                            cron 表達式為標準 5 欄位格式（分 時 日 月 週）。\
                            executor 可選 worker（快速）、llm（主模型）、brain（最強）。\
                            完成後用簡短中文回報結果。";
                        let result = api::chat_with_tools(worker.as_ref(), system, &args, &tools, None).await?;
                        // NLP 路徑後重新同步：reload 所有 cron
                        if let Ok(config) = tool::cron::load_config(&cp) {
                            let running = pool.running_ids();
                            for job in config.jobs.iter().filter(|j| j.enabled) {
                                if !running.contains(&job.id) {
                                    pool.spawn_cron(job);
                                }
                            }
                        }
                        Ok(result.text)
                    })
                }),
            );

            // /task 指令
            let tp = tasks_path.clone();
            let task_worker = Arc::clone(&res.worker);
            let task_pool = Arc::clone(&pool);
            cmd_registry.register(
                "task",
                "管理一次性排程任務（支援自然語言）",
                "/task [list|add|remove|<自然語言>]",
                Arc::new(move |_sender, args| {
                    let tp = tp.clone();
                    let worker = Arc::clone(&task_worker);
                    let pool = Arc::clone(&task_pool);
                    Box::pin(async move {
                        let result = tool::task::handle_slash_command(&tp, &args);
                        match result {
                            Ok(ref text) if text.starts_with("用法:") => {}
                            other => {
                                let trimmed = args.trim();
                                // 新增時同步 spawn
                                if trimmed.starts_with("add ")
                                    && let Some(task) = tool::task::load_config(&tp)
                                        .ok()
                                        .and_then(|c| c.tasks.last().cloned())
                                {
                                    pool.spawn_task(&task);
                                }
                                // 刪除時中止
                                if (trimmed.starts_with("remove ") || trimmed.starts_with("rm "))
                                    && let Ok(text) = &other
                                    && let Some(id) = extract_removed_id(text, "一次性任務")
                                {
                                    pool.abort(&id);
                                }
                                return other;
                            }
                        }
                        // fallback: 用 worker + task 工具解讀自然語言
                        let mut tools = tool::ToolRegistry::new();
                        for t in tool::task::tools(tp.parent().unwrap().to_path_buf()) {
                            tools.register(t);
                        }
                        let system = "你是一個排程任務管理助手。\
                            使用者想管理一次性排程任務，請根據使用者的描述使用工具完成操作。\
                            時間格式支援 ISO 8601 或 YYYY-MM-DD HH:MM。\
                            executor 可選 worker（快速）、llm（主模型）、brain（最強）。\
                            完成後用簡短中文回報結果。";
                        let result = api::chat_with_tools(worker.as_ref(), system, &args, &tools, None).await?;
                        // NLP 路徑後重新同步
                        if let Ok(config) = tool::task::load_config(&tp) {
                            let running = pool.running_ids();
                            for task in &config.tasks {
                                if !running.contains(&task.id) {
                                    pool.spawn_task(task);
                                }
                            }
                        }
                        Ok(result.text)
                    })
                }),
            );
        }

        // /skills 指令：列出可用 skills
        if let Some(ref sp) = skills_path {
            let sp_clone = sp.clone();
            cmd_registry.register(
                "skills",
                "列出可用的 skills",
                "/skills",
                Arc::new(move |_sender, _args| {
                    let sp = sp_clone.clone();
                    Box::pin(async move {
                        let entries = tool::skills::list_skill_entries(&sp);
                        if entries.is_empty() {
                            return Ok("目前沒有可用的 skills。".into());
                        }
                        let list: Vec<String> = entries
                            .iter()
                            .map(|(name, desc)| {
                                if desc.is_empty() {
                                    format!("• {name}")
                                } else {
                                    format!("• {name} — {desc}")
                                }
                            })
                            .collect();
                        Ok(format!("可用 Skills:\n{}", list.join("\n")))
                    })
                }),
            );
        }

        // ingest 模式：註冊 /ask 指令，用於查詢向量資料庫
        if agent_mode == AgentMode::Ingest {
            let ask_ctx = Arc::new(MessageContext {
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
            });
            cmd_registry.register(
                "ask",
                "查詢已收集的資料",
                "/ask <問題>",
                Arc::new(move |_sender, args| {
                    let ctx = Arc::clone(&ask_ctx);
                    Box::pin(async move {
                        let question = args.trim();
                        if question.is_empty() {
                            return Ok("用法: /ask <問題>".into());
                        }
                        handle_query(&ctx, question, &[], None).await
                    })
                }),
            );
        }

        // LLM 互動指令（來自 cmd.toml）
        for cmd_def in &agent_llm_commands {
            let cmd_prompt = cmd_def.prompt.clone();
            let cmd_tools = cmd_def.tools.clone();
            let llm_for_cmd = Arc::clone(&res.llm);
            let pg_for_cmd = Arc::clone(&res.pg);
            let soul_for_cmd = agent_soul.clone();
            let prompt_for_cmd = agent_prompt.clone();
            let aid_for_cmd = agent_id.clone();

            cmd_registry.register(
                cmd_def.name.clone(),
                cmd_def.description.clone(),
                format!("/{} [補充說明]", cmd_def.name),
                Arc::new(move |_sender, args| {
                    let cmd_prompt = cmd_prompt.clone();
                    let cmd_tools = cmd_tools.clone();
                    let llm = Arc::clone(&llm_for_cmd);
                    let pg = Arc::clone(&pg_for_cmd);
                    let soul = soul_for_cmd.clone();
                    let prompt = prompt_for_cmd.clone();
                    let aid = aid_for_cmd.clone();
                    Box::pin(async move {
                        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %:z");

                        let soul_section = if soul.is_empty() {
                            String::new()
                        } else {
                            format!("## 性格設定\n{soul}\n\n")
                        };

                        let system = format!(
                            "你是一個 AI 助手。\n\n\
                             目前時間：{now}\n\n\
                             {soul_section}\
                             {prompt}"
                        );

                        let user_message = if args.trim().is_empty() {
                            cmd_prompt
                        } else {
                            format!("{cmd_prompt}\n\n{}", args.trim())
                        };

                        let result = if cmd_tools.is_empty() {
                            llm.generate(api::GenerateParams {
                                system: Some(&system),
                                user_message: &user_message,
                                images: &[],
                                json_mode: false,
                                temperature: 0.7,
                            })
                            .await?
                        } else {
                            let tools = tool::build_registry(&cmd_tools, &pg, &aid);
                            api::chat_with_tools(llm.as_ref(), &system, &user_message, &tools, None)
                                .await?
                        };

                        Ok(result.text)
                    })
                }),
            );
        }

        // help 最後註冊，help_text() 會包含前面所有指令 + help 自己
        cmd_registry.register(
            "help",
            "顯示可用指令列表",
            "/help",
            Arc::new(|_sender, _args| Box::pin(async { Ok(String::new()) })),
        );
        let help_text = cmd_registry.help_text();
        cmd_registry.register(
            "help",
            "顯示可用指令列表",
            "/help",
            Arc::new(move |_sender, _args| {
                let text = help_text.clone();
                Box::pin(async move { Ok(text) })
            }),
        );

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
                                let (file_ids, source_chat_id, source_message_id) = extract_image_meta(&msg.images);
                                let image_meta = ImageMeta { file_ids: &file_ids, source_chat_id, source_message_id };
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

/// 從移除結果文字中提取 id（格式：「已移除XXX「id」」）
fn extract_removed_id(text: &str, _kind: &str) -> Option<String> {
    // 匹配「xxx」中的 xxx
    let start = text.find('「')? + '「'.len_utf8();
    let end = text[start..].find('」')? + start;
    Some(text[start..end].to_owned())
}

/// 註冊心跳查看/編輯指令
fn register_heartbeat_commands(registry: &mut CommandRegistry, apath: &std::path::Path) {
    let hb_path = apath.join("HEARTBEAT.md");
    registry.register(
        "heartbeat",
        "查看或設定心跳提示詞",
        "/heartbeat [show|set <內容>]",
        Arc::new(move |_sender, args| {
            let hb_path = hb_path.clone();
            Box::pin(async move {
                let subcmd = args.trim();
                if subcmd.is_empty() || subcmd == "show" {
                    let content = tokio::fs::read_to_string(&hb_path)
                        .await
                        .unwrap_or_else(|_| "（無 HEARTBEAT.md）".into());
                    Ok(format!("HEARTBEAT.md:\n\n{content}"))
                } else if let Some(new_content) = subcmd.strip_prefix("set ") {
                    tokio::fs::write(&hb_path, new_content).await?;
                    Ok("HEARTBEAT.md 已更新".into())
                } else {
                    Ok("用法: /heartbeat [show] | /heartbeat set <內容>".into())
                }
            })
        }),
    );

    let bhb_path = apath.join("BRAIN_HEARTBEAT.md");
    registry.register(
        "brain_heartbeat",
        "查看或設定大腦心跳提示詞",
        "/brain_heartbeat [show|set <內容>]",
        Arc::new(move |_sender, args| {
            let bhb_path = bhb_path.clone();
            Box::pin(async move {
                let subcmd = args.trim();
                if subcmd.is_empty() || subcmd == "show" {
                    let content = tokio::fs::read_to_string(&bhb_path)
                        .await
                        .unwrap_or_else(|_| "（無 BRAIN_HEARTBEAT.md）".into());
                    Ok(format!("BRAIN_HEARTBEAT.md:\n\n{content}"))
                } else if let Some(new_content) = subcmd.strip_prefix("set ") {
                    tokio::fs::write(&bhb_path, new_content).await?;
                    Ok("BRAIN_HEARTBEAT.md 已更新".into())
                } else {
                    Ok("用法: /brain_heartbeat [show] | /brain_heartbeat set <內容>".into())
                }
            })
        }),
    );
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
fn extract_image_meta(images: &[crate::channel::ImageData]) -> (Vec<String>, Option<i64>, Option<i32>) {
    let file_ids: Vec<String> = images.iter().filter_map(|img| img.file_id.clone()).collect();
    // 取第一張圖片的來源資訊（同一則訊息的圖片來自同一個 chat/message）
    let source_chat_id = images.first().and_then(|img| img.source_chat_id);
    let source_message_id = images.first().and_then(|img| img.source_message_id);
    (file_ids, source_chat_id, source_message_id)
}

/// 僅入庫（ingest 模式），不呼叫 LLM 回覆
async fn handle_ingest(ctx: &MessageContext, text: &str, images: &[api::ImageInput], image_meta: &ImageMeta<'_>) -> Result<()> {
    let embed_images = if ctx.image_embed { images } else { &[] };
    let (_, wk_usage) =
        search::ingest(&ctx.pg, &ctx.qdrant, ctx.embedder.as_ref(), ctx.worker.as_ref(), &ctx.agent_id, text, embed_images, image_meta).await?;

    db::postgres::insert_token_usage(&ctx.pg, &ctx.agent_id, &ctx.worker_model, wk_usage.input_tokens, wk_usage.output_tokens).await?;

    Ok(())
}

/// 查詢（不入庫）：檢索記憶 + LLM 回答
async fn handle_query(ctx: &MessageContext, question: &str, images: &[api::ImageInput], progress: Option<&api::ProgressSender>) -> Result<String> {
    let memo = db::postgres::get_scratchpad(&ctx.pg, &ctx.agent_id)
        .await?
        .unwrap_or_default();

    let results = search::retrieve(&ctx.pg, &ctx.qdrant, ctx.embedder.as_ref(), &ctx.agent_id, question, 5).await?;

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

    let memo_section = if memo.is_empty() { "（空）".to_owned() } else { memo };
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

    let result = api::chat_with_images(ctx.llm.as_ref(), &system, question, images, &tools, progress).await?;

    db::postgres::insert_token_usage(&ctx.pg, &ctx.agent_id, &ctx.llm_model, result.input_tokens, result.output_tokens).await?;

    Ok(result.text)
}

/// 處理單則訊息（chat 模式：先回覆，再入庫使用者訊息與 LLM 回覆）
async fn handle_message(
    ctx: &MessageContext,
    msg: &crate::channel::IncomingMessage,
) -> Result<Option<String>> {
    let images = to_image_inputs(&msg.images);
    let (file_ids, source_chat_id, source_message_id) = extract_image_meta(&msg.images);
    let image_meta = ImageMeta { file_ids: &file_ids, source_chat_id, source_message_id };

    // 1. 先查詢 + 回覆（LLM chat 端一律傳圖，不受 image_embed 影響）
    let reply = handle_query(ctx, &msg.text, &images, msg.progress.as_ref()).await?;

    // 2. 入庫使用者訊息（image_embed 由 handle_ingest 內部判斷）
    let empty_meta = ImageMeta { file_ids: &[], source_chat_id: None, source_message_id: None };
    handle_ingest(ctx, &msg.text, &images, &image_meta).await?;

    // 3. 入庫 LLM 回覆
    handle_ingest(ctx, &reply, &[], &empty_meta).await?;

    Ok(Some(reply))
}
