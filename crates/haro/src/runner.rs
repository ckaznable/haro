use std::str::FromStr;
use std::sync::Arc;

use anyhow::Result;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

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
            handles.push(tokio::spawn(run_heartbeat(HeartbeatTask {
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
            handles.push(tokio::spawn(run_heartbeat(HeartbeatTask {
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

        // 心跳查看/編輯指令
        if let Some(ref apath) = agent_path {
            register_heartbeat_commands(&mut cmd_registry, apath);
        }

        // Skills 目錄（存在才啟用）
        let skills_path = agent_path
            .as_ref()
            .map(|p| p.join("skills"))
            .filter(|p| p.is_dir());

        // /cron 指令 + 背景 cron runner
        if let Some(ref apath) = agent_path {
            let cron_path = apath.join("cron.toml");
            let cp = cron_path.clone();
            let cron_worker = Arc::clone(&res.worker);
            cmd_registry.register(
                "cron",
                "管理排程任務（支援自然語言）",
                "/cron [list|add|remove|enable|disable|<自然語言>]",
                Arc::new(move |_sender, args| {
                    let cp = cp.clone();
                    let worker = Arc::clone(&cron_worker);
                    Box::pin(async move {
                        // 先嘗試結構化指令
                        let result = tool::cron::handle_slash_command(&cp, &args);
                        match result {
                            // 回傳「用法」表示格式不對，改用 worker 解讀
                            Ok(ref text) if text.starts_with("用法:") => {}
                            other => return other,
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
                        Ok(result.text)
                    })
                }),
            );

            // /task 指令
            let tasks_path = apath.join("tasks.toml");
            let tp = tasks_path.clone();
            let task_worker = Arc::clone(&res.worker);
            cmd_registry.register(
                "task",
                "管理一次性排程任務（支援自然語言）",
                "/task [list|add|remove|<自然語言>]",
                Arc::new(move |_sender, args| {
                    let tp = tp.clone();
                    let worker = Arc::clone(&task_worker);
                    Box::pin(async move {
                        let result = tool::task::handle_slash_command(&tp, &args);
                        match result {
                            Ok(ref text) if text.starts_with("用法:") => {}
                            other => return other,
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
                        Ok(result.text)
                    })
                }),
            );

            info!(agent_id = %agent_id, "啟動 Cron 排程任務");
            handles.push(tokio::spawn(run_cron(CronRunnerTask {
                pg: Arc::clone(&res.pg),
                worker: Arc::clone(&res.worker),
                llm: Arc::clone(&res.llm),
                brain: Arc::clone(&res.brain),
                agent_id: agent_id.clone(),
                agent_prompt: agent_prompt.clone(),
                cron_path,
                tasks_path,
                llm_model: cfg.llm.model.clone(),
                brain_model: brain_model.clone(),
                worker_model: cfg.worker.model.clone(),
                notifiers: notifiers.clone(),
                searxng_url: cfg.searxng_url.clone(),
                skills_path: skills_path.clone(),
            })));
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
         ## 相關資料\n{context_str}{pending_section}\n\n\
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

const HEARTBEAT_TRIAGE_PROMPT: &str = "\
You are a triage agent. Given the heartbeat instructions, current time, and scratchpad context, \
decide whether there is a task that needs to be executed RIGHT NOW.\n\
Consider time-sensitive tasks, scheduled actions, and pending items.\n\n\
Reply with ONLY \"yes\" (needs action now) or \"no\" (skip this cycle).";

/// 心跳任務設定
struct HeartbeatTask {
    pg: Arc<sqlx::PgPool>,
    worker: Arc<api::gemini::GeminiProvider>,
    llm: Arc<api::gemini::GeminiProvider>,
    agent_id: String,
    agent_prompt: String,
    heartbeat_prompt: String,
    interval_secs: u64,
    llm_model: String,
    notifiers: Vec<Arc<dyn Notifier>>,
    searxng_url: Option<String>,
    agent_path: Option<std::path::PathBuf>,
}

/// 心跳任務：定期喚醒 worker 判斷是否需要呼叫 LLM 執行任務
async fn run_heartbeat(task: HeartbeatTask) -> Result<()> {
    let interval = std::time::Duration::from_secs(task.interval_secs);

    loop {
        tokio::time::sleep(interval).await;

        info!(agent_id = %task.agent_id, "心跳觸發");

        let memo = db::postgres::get_scratchpad(&task.pg, &task.agent_id)
            .await
            .unwrap_or(None)
            .unwrap_or_default();

        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %:z");

        let context = format!(
            "目前時間：{now}\n\n\
             ## 心跳指示\n{}\n\n\
             ## 備忘錄\n{memo}",
            task.heartbeat_prompt
        );

        let triage_result = task
            .worker
            .generate(api::GenerateParams {
                system: Some(HEARTBEAT_TRIAGE_PROMPT),
                user_message: &context,
                images: &[],
                json_mode: false,
                temperature: 0.0,
            })
            .await;

        let needs_action = match triage_result {
            Ok(r) => r.text.trim().to_lowercase().starts_with("yes"),
            Err(e) => {
                error!(agent_id = %task.agent_id, "心跳分流判斷失敗: {e:#}");
                continue;
            }
        };

        if !needs_action {
            info!(agent_id = %task.agent_id, "心跳：無需執行任務");
            continue;
        }

        info!(agent_id = %task.agent_id, "心跳：喚醒 LLM 執行任務");

        let mut tools = tool::ToolRegistry::new();
        for t in tool::scratchpad::tools(Arc::clone(&task.pg), task.agent_id.clone()) {
            tools.register(t);
        }
        tools.register(tool::fetch::tool());
        if task.llm.grounding() {
            tools.register(tool::search::tool(Arc::clone(&task.llm)));
        }
        if let Some(ref url) = task.searxng_url {
            tools.register(tool::searxng::tool(url.clone()));
        }
        if let Some(ref ap) = task.agent_path {
            for t in tool::heartbeat::tools(ap.clone()) {
                tools.register(t);
            }
        }
        if !task.notifiers.is_empty() {
            tools.register(tool::notify::tool(task.notifiers.clone()));
        }

        let notify_instruction = if task.notifiers.is_empty() {
            ""
        } else {
            "\n如果需要主動通知使用者，請使用 send_message 工具發送訊息到頻道。"
        };

        let system = format!(
            "你是一個具有長期記憶的 AI 助手。\n\n\
             目前時間：{now}\n\n\
             {}\n\n\
             ## 備忘錄 (Scratchpad)\n{memo}\n\n\
             ## 指示\n\
             這是一個定期心跳喚醒。請根據以下心跳指示執行任務。\n\
             如果有重要的事情需要記住，請使用 save_memo 工具儲存到備忘錄中。\n\
             備忘錄只用於紀錄固定且明確的資訊（例如偏好、規則、待辦事項），不要紀錄曾經做過的事情。\n\
             需要回報的事項必須使用 send_message 工具發送到頻道，不要只在內部處理。{notify_instruction}\n\n\
             ## 心跳指示\n{}",
            task.agent_prompt, task.heartbeat_prompt
        );

        let result = api::chat_with_tools(task.llm.as_ref(), &system, "心跳觸發，請執行任務。", &tools, None).await;

        match result {
            Ok(r) => {
                let _ = db::postgres::insert_token_usage(
                    &task.pg,
                    &task.agent_id,
                    &task.llm_model,
                    r.input_tokens,
                    r.output_tokens,
                )
                .await;
                info!(agent_id = %task.agent_id, "心跳任務完成: {}", r.text.chars().take(100).collect::<String>());
            }
            Err(e) => {
                error!(agent_id = %task.agent_id, "心跳執行失敗: {e:#}");
            }
        }
    }
}

// ── Cron 排程任務 ──

/// Cron 排程任務設定
struct CronRunnerTask {
    pg: Arc<sqlx::PgPool>,
    worker: Arc<api::gemini::GeminiProvider>,
    llm: Arc<api::gemini::GeminiProvider>,
    brain: Arc<api::gemini::GeminiProvider>,
    agent_id: String,
    agent_prompt: String,
    cron_path: std::path::PathBuf,
    tasks_path: std::path::PathBuf,
    llm_model: String,
    brain_model: String,
    worker_model: String,
    notifiers: Vec<Arc<dyn Notifier>>,
    searxng_url: Option<String>,
    skills_path: Option<std::path::PathBuf>,
}

/// 下一個要執行的事件類型
enum NextEvent {
    /// 週期性 cron job
    CronJob { id: String },
    /// 一次性排程任務
    ScheduledTask { id: String },
}

/// Cron 排程背景迴圈：載入 cron.toml，計算下次觸發時間，到時執行
async fn run_cron(task: CronRunnerTask) -> Result<()> {
    use chrono::Local;
    use tool::cron::{load_config, Executor};

    // 初始等待 10 秒，讓其他服務先啟動
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    loop {
        // 每次迴圈重新載入 cron.toml（支援熱更新）
        let config = match load_config(&task.cron_path) {
            Ok(c) => c,
            Err(e) => {
                error!(agent_id = %task.agent_id, "載入 cron.toml 失敗: {e:#}");
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                continue;
            }
        };

        let now = Local::now();
        let mut earliest_event: Option<NextEvent> = None;
        let mut earliest_time: Option<chrono::DateTime<Local>> = None;

        // 檢查 cron jobs
        for job in config.jobs.iter().filter(|j| j.enabled) {
            let full_expr = format!("0 {} *", job.cron);
            let schedule = match cron::Schedule::from_str(&full_expr) {
                Ok(s) => s,
                Err(e) => {
                    warn!(agent_id = %task.agent_id, job_id = %job.id, "cron 表達式無效，跳過: {e}");
                    continue;
                }
            };

            if let Some(next) = schedule.upcoming(Local).next() {
                if earliest_time.is_none() || next < earliest_time.unwrap() {
                    earliest_time = Some(next);
                    earliest_event = Some(NextEvent::CronJob { id: job.id.clone() });
                }
            }
        }

        // 檢查一次性排程任務（tasks.toml）
        let tasks_config = tool::task::load_config(&task.tasks_path).unwrap_or_default();
        for st in &tasks_config.tasks {
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&st.run_at) {
                let dt_local = dt.with_timezone(&Local);
                if dt_local <= now {
                    // 已過期但未執行，立即執行
                    earliest_time = Some(now);
                    earliest_event = Some(NextEvent::ScheduledTask { id: st.id.clone() });
                    break; // 過期任務優先
                } else if earliest_time.is_none() || dt_local < earliest_time.unwrap() {
                    earliest_time = Some(dt_local);
                    earliest_event = Some(NextEvent::ScheduledTask { id: st.id.clone() });
                }
            } else {
                warn!(agent_id = %task.agent_id, task_id = %st.id, "一次性任務時間格式無效，跳過");
            }
        }

        let (Some(next_time), Some(event)) = (earliest_time, earliest_event) else {
            // 沒有任何排程，等 60 秒後再檢查
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            continue;
        };

        let event_id = match &event {
            NextEvent::CronJob { id } => id.as_str(),
            NextEvent::ScheduledTask { id } => id.as_str(),
        };

        let wait_duration = (next_time - now).to_std().unwrap_or(std::time::Duration::ZERO);
        if !wait_duration.is_zero() {
            info!(
                agent_id = %task.agent_id,
                next_event = %event_id,
                next_time = %next_time.format("%H:%M:%S"),
                wait_secs = wait_duration.as_secs(),
                "排程等待下次觸發"
            );
            tokio::time::sleep(wait_duration).await;
        }

        // 重新載入設定（等待期間可能有變更）
        let config = match load_config(&task.cron_path) {
            Ok(c) => c,
            Err(e) => {
                error!(agent_id = %task.agent_id, "載入 cron.toml 失敗: {e:#}");
                continue;
            }
        };

        // 取得 executor + prompt
        let (executor_type, prompt_text, event_label) = match &event {
            NextEvent::CronJob { id } => {
                let Some(job) = config.jobs.iter().find(|j| j.id == *id && j.enabled) else {
                    info!(agent_id = %task.agent_id, job_id = %id, "任務已被移除或停用，跳過");
                    continue;
                };
                (job.executor.clone(), job.prompt.clone(), format!("cron「{id}」"))
            }
            NextEvent::ScheduledTask { id } => {
                let tasks_config = tool::task::load_config(&task.tasks_path).unwrap_or_default();
                let Some(st) = tasks_config.tasks.iter().find(|t| t.id == *id) else {
                    info!(agent_id = %task.agent_id, task_id = %id, "任務已被移除，跳過");
                    continue;
                };
                (st.executor.clone(), st.prompt.clone(), format!("一次性任務「{id}」"))
            }
        };

        info!(agent_id = %task.agent_id, event = %event_label, executor = %executor_type, "觸發排程任務");

        let (executor, model_name): (&api::gemini::GeminiProvider, &str) = match executor_type {
            Executor::Worker => (task.worker.as_ref(), &task.worker_model),
            Executor::Llm => (task.llm.as_ref(), &task.llm_model),
            Executor::Brain => (task.brain.as_ref(), &task.brain_model),
        };

        // 建立工具
        let mut tools = tool::ToolRegistry::new();
        for t in tool::scratchpad::tools(Arc::clone(&task.pg), task.agent_id.clone()) {
            tools.register(t);
        }
        tools.register(tool::fetch::tool());
        if executor.grounding() {
            let executor_arc = match executor_type {
                Executor::Worker => Arc::clone(&task.worker),
                Executor::Llm => Arc::clone(&task.llm),
                Executor::Brain => Arc::clone(&task.brain),
            };
            tools.register(tool::search::tool(executor_arc));
        }
        if let Some(ref url) = task.searxng_url {
            tools.register(tool::searxng::tool(url.clone()));
        }
        if let Some(ref sp) = task.skills_path {
            for t in tool::skills::tools(sp.clone()) {
                tools.register(t);
            }
        }
        if !task.notifiers.is_empty() {
            tools.register(tool::notify::tool(task.notifiers.clone()));
        }

        let memo = db::postgres::get_scratchpad(&task.pg, &task.agent_id)
            .await
            .unwrap_or(None)
            .unwrap_or_default();

        let now = Local::now().format("%Y-%m-%d %H:%M:%S %:z");
        let notify_instruction = if task.notifiers.is_empty() {
            ""
        } else {
            "\n如果需要回報結果，請使用 send_message 工具發送到頻道。"
        };

        let system = format!(
            "你是一個 AI 助手，正在執行排程任務。\n\n\
             目前時間：{now}\n\n\
             {}\n\n\
             ## 備忘錄 (Scratchpad)\n{memo}\n\n\
             ## 指示\n\
             這是一個排程任務，請根據以下提示詞執行。\n\
             如果有重要的事情需要記住，請使用 save_memo 工具儲存。\n\
             需要回報的事項必須使用 send_message 工具發送到頻道。{notify_instruction}",
            task.agent_prompt
        );

        let user_message = format!("{event_label}觸發，請執行：\n\n{prompt_text}");

        let result = api::chat_with_tools(executor, &system, &user_message, &tools, None).await;

        match result {
            Ok(r) => {
                let _ = db::postgres::insert_token_usage(
                    &task.pg,
                    &task.agent_id,
                    model_name,
                    r.input_tokens,
                    r.output_tokens,
                )
                .await;
                info!(agent_id = %task.agent_id, event = %event_label, "排程任務完成");
            }
            Err(e) => {
                error!(agent_id = %task.agent_id, event = %event_label, "排程任務失敗: {e:#}");
            }
        }

        // 一次性任務完成後從 tasks.toml 中刪除
        if let NextEvent::ScheduledTask { id } = &event {
            if let Err(e) = tool::task::remove_task(&task.tasks_path, id) {
                error!(agent_id = %task.agent_id, task_id = %id, "刪除已完成任務失敗: {e:#}");
            }
        }
    }
}
