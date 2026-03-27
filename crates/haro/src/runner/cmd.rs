use std::sync::Arc;

use crate::agent::{AgentMode, LlmCmdDef};
use crate::api::{self, LlmProvider};
use crate::channel::{CommandRegistry, Notifier};
use crate::config::AppConfig;
use crate::{db, tool};

use super::scheduler;
use super::{handle_query, MessageContext};

/// 註冊所有 slash 指令所需的上下文
pub(crate) struct CmdContext<'a> {
    pub res: &'a super::SharedResources,
    pub cfg: &'a AppConfig,
    pub agent_id: &'a str,
    pub agent_mode: &'a AgentMode,
    pub agent_prompt: &'a str,
    pub agent_soul: &'a str,
    pub agent_path: &'a Option<std::path::PathBuf>,
    pub skills_path: &'a Option<std::path::PathBuf>,
    pub notifiers: &'a [Arc<dyn Notifier>],
    pub llm_commands: &'a [LlmCmdDef],
    pub brain_model: &'a str,
}

/// 建立指令註冊表並註冊所有 slash 指令，回傳 (registry, handles)
pub(crate) fn register_all(ctx: &CmdContext<'_>) -> CommandRegistry {
    let mut reg = CommandRegistry::new();

    register_basic(&mut reg);
    register_memo(&mut reg, ctx);
    register_heartbeat(&mut reg, ctx);
    register_scheduler(&mut reg, ctx);
    register_skills(&mut reg, ctx);
    register_ask(&mut reg, ctx);
    register_llm_commands(&mut reg, ctx);
    register_tools(&mut reg, ctx);
    register_help(&mut reg);

    reg
}

// ── 基本指令 ──

fn register_basic(reg: &mut CommandRegistry) {
    reg.register(
        "stop",
        "中斷正在處理的任務",
        "/stop",
        Arc::new(|_sender, _args| Box::pin(async { Ok("此指令由頻道直接處理。".into()) })),
    );

    reg.register(
        "ping",
        "檢查機器人是否在線",
        "/ping",
        Arc::new(|_sender, _args| Box::pin(async { Ok("OK".into()) })),
    );
}

// ── /memo ──

fn register_memo(reg: &mut CommandRegistry, ctx: &CmdContext<'_>) {
    let pg = Arc::clone(&ctx.res.pg);
    let agent_id = ctx.agent_id.to_owned();
    reg.register(
        "memo",
        "查看或編輯備忘錄",
        "/memo [set <內容>]",
        Arc::new(move |_sender, args| {
            let pg = Arc::clone(&pg);
            let aid = agent_id.clone();
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

// ── /heartbeat, /brain_heartbeat ──

fn register_heartbeat(reg: &mut CommandRegistry, ctx: &CmdContext<'_>) {
    let Some(apath) = ctx.agent_path else { return };

    let hb_path = apath.join("HEARTBEAT.md");
    reg.register(
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
    reg.register(
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

// ── /cron, /task + 排程器啟動 ──

fn register_scheduler(reg: &mut CommandRegistry, ctx: &CmdContext<'_>) {
    let Some(apath) = ctx.agent_path else { return };

    let cron_path = apath.join("cron.toml");
    let tasks_path = apath.join("tasks.toml");

    let pool = scheduler::RunningPool::new(scheduler::SchedulerResources {
        pg: Arc::clone(&ctx.res.pg),
        worker: Arc::clone(&ctx.res.worker),
        llm: Arc::clone(&ctx.res.llm),
        brain: Arc::clone(&ctx.res.brain),
        agent_id: ctx.agent_id.to_owned(),
        agent_prompt: ctx.agent_prompt.to_owned(),
        tasks_path: tasks_path.clone(),
        llm_model: ctx.cfg.llm.model.clone(),
        brain_model: ctx.brain_model.to_owned(),
        worker_model: ctx.cfg.worker.model.clone(),
        notifiers: ctx.notifiers.to_vec(),
        searxng_url: ctx.cfg.searxng_url.clone(),
        skills_path: ctx.skills_path.clone(),
    });

    tracing::info!(agent_id = %ctx.agent_id, "啟動排程任務");
    pool.spawn_all(&cron_path, &tasks_path);

    // /cron
    {
        let cp = cron_path.clone();
        let worker = Arc::clone(&ctx.res.worker);
        let pool = Arc::clone(&pool);
        reg.register(
            "cron",
            "管理排程任務（支援自然語言）",
            "/cron [list|add|remove|enable|disable|<自然語言>]",
            Arc::new(move |_sender, args| {
                let cp = cp.clone();
                let worker = Arc::clone(&worker);
                let pool = Arc::clone(&pool);
                Box::pin(async move {
                    let result = tool::cron::handle_slash_command(&cp, &args);
                    match result {
                        Ok(ref text) if text.starts_with("用法:") => {}
                        other => {
                            let trimmed = args.trim();
                            if trimmed.starts_with("add ")
                                && let Some(job) = tool::cron::load_config(&cp)
                                    .ok()
                                    .and_then(|c| c.jobs.last().cloned())
                            {
                                pool.spawn_cron(&job);
                            }
                            if (trimmed.starts_with("remove ") || trimmed.starts_with("rm "))
                                && let Ok(text) = &other
                                && let Some(id) = extract_removed_id(text, "排程任務")
                            {
                                pool.abort(&id);
                            }
                            return other;
                        }
                    }
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
    }

    // /task
    {
        let tp = tasks_path.clone();
        let worker = Arc::clone(&ctx.res.worker);
        let pool = Arc::clone(&pool);
        reg.register(
            "task",
            "管理一次性排程任務（支援自然語言）",
            "/task [list|add|remove|<自然語言>]",
            Arc::new(move |_sender, args| {
                let tp = tp.clone();
                let worker = Arc::clone(&worker);
                let pool = Arc::clone(&pool);
                Box::pin(async move {
                    let result = tool::task::handle_slash_command(&tp, &args);
                    match result {
                        Ok(ref text) if text.starts_with("用法:") => {}
                        other => {
                            let trimmed = args.trim();
                            if trimmed.starts_with("add ")
                                && let Some(task) = tool::task::load_config(&tp)
                                    .ok()
                                    .and_then(|c| c.tasks.last().cloned())
                            {
                                pool.spawn_task(&task);
                            }
                            if (trimmed.starts_with("remove ") || trimmed.starts_with("rm "))
                                && let Ok(text) = &other
                                && let Some(id) = extract_removed_id(text, "一次性任務")
                            {
                                pool.abort(&id);
                            }
                            return other;
                        }
                    }
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
}

// ── /skills ──

fn register_skills(reg: &mut CommandRegistry, ctx: &CmdContext<'_>) {
    let Some(sp) = ctx.skills_path else { return };
    let sp = sp.clone();
    reg.register(
        "skills",
        "列出可用的 skills",
        "/skills",
        Arc::new(move |_sender, _args| {
            let sp = sp.clone();
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

// ── /ask (ingest 模式) ──

fn register_ask(reg: &mut CommandRegistry, ctx: &CmdContext<'_>) {
    if *ctx.agent_mode != AgentMode::Ingest {
        return;
    }
    let ask_ctx = Arc::new(MessageContext {
        pg: Arc::clone(&ctx.res.pg),
        qdrant: Arc::clone(&ctx.res.qdrant),
        embedder: Arc::clone(&ctx.res.embedder),
        llm: Arc::clone(&ctx.res.llm),
        worker: Arc::clone(&ctx.res.worker),
        agent_id: ctx.agent_id.to_owned(),
        prompt: ctx.agent_prompt.to_owned(),
        soul: ctx.agent_soul.to_owned(),
        worker_model: ctx.cfg.worker.model.clone(),
        llm_model: ctx.cfg.llm.model.clone(),
        image_embed: ctx.cfg.embedding.image_embed,
        skills_path: ctx.skills_path.clone(),
        agent_path: ctx.agent_path.clone(),
        searxng_url: ctx.cfg.searxng_url.clone(),
    });
    reg.register(
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

// ── LLM 互動指令（cmd.toml）──

fn register_llm_commands(reg: &mut CommandRegistry, ctx: &CmdContext<'_>) {
    for cmd_def in ctx.llm_commands {
        let cmd_prompt = cmd_def.prompt.clone();
        let cmd_tools = cmd_def.tools.clone();
        let llm = Arc::clone(&ctx.res.llm);
        let pg = Arc::clone(&ctx.res.pg);
        let soul = ctx.agent_soul.to_owned();
        let prompt = ctx.agent_prompt.to_owned();
        let aid = ctx.agent_id.to_owned();

        reg.register(
            cmd_def.name.clone(),
            cmd_def.description.clone(),
            format!("/{} [補充說明]", cmd_def.name),
            Arc::new(move |_sender, args| {
                let cmd_prompt = cmd_prompt.clone();
                let cmd_tools = cmd_tools.clone();
                let llm = Arc::clone(&llm);
                let pg = Arc::clone(&pg);
                let soul = soul.clone();
                let prompt = prompt.clone();
                let aid = aid.clone();
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
}

// ── /tools ──

fn register_tools(reg: &mut CommandRegistry, ctx: &CmdContext<'_>) {
    let pg = Arc::clone(&ctx.res.pg);
    let llm = Arc::clone(&ctx.res.llm);
    let agent_id = ctx.agent_id.to_owned();
    let skills_path = ctx.skills_path.clone();
    let agent_path = ctx.agent_path.clone();
    let searxng_url = ctx.cfg.searxng_url.clone();
    let notifiers: Vec<Arc<dyn Notifier>> = ctx.notifiers.to_vec();
    reg.register(
        "tools",
        "列出 LLM 可使用的工具",
        "/tools",
        Arc::new(move |_sender, _args| {
            let pg = Arc::clone(&pg);
            let llm = Arc::clone(&llm);
            let aid = agent_id.clone();
            let sp = skills_path.clone();
            let ap = agent_path.clone();
            let searxng = searxng_url.clone();
            let notifiers = notifiers.clone();
            Box::pin(async move {
                let mut registry = tool::ToolRegistry::new();
                for t in tool::scratchpad::tools(pg, aid) {
                    registry.register(t);
                }
                registry.register(tool::fetch::tool());
                if llm.grounding() {
                    registry.register(tool::search::tool(llm));
                }
                if let Some(ref url) = searxng {
                    registry.register(tool::searxng::tool(url.clone()));
                }
                if let Some(ref ap) = ap {
                    for t in tool::heartbeat::tools(ap.clone()) {
                        registry.register(t);
                    }
                    for t in tool::cron::tools(ap.clone()) {
                        registry.register(t);
                    }
                    for t in tool::task::tools(ap.clone()) {
                        registry.register(t);
                    }
                }
                if let Some(ref sp) = sp {
                    for t in tool::skills::tools(sp.clone()) {
                        registry.register(t);
                    }
                }
                if !notifiers.is_empty() {
                    registry.register(tool::notify::tool(notifiers));
                }

                let defs = registry.definitions();
                if defs.is_empty() {
                    return Ok("LLM 目前沒有可用的工具。".into());
                }
                let list: Vec<String> = defs
                    .iter()
                    .map(|d| format!("• {} — {}", d.name, d.description))
                    .collect();
                Ok(format!("LLM 可用工具（共 {} 個）：\n{}", defs.len(), list.join("\n")))
            })
        }),
    );
}

// ── /help（最後註冊）──

fn register_help(reg: &mut CommandRegistry) {
    reg.register(
        "help",
        "顯示可用指令列表",
        "/help",
        Arc::new(|_sender, _args| Box::pin(async { Ok(String::new()) })),
    );
    let help_text = reg.help_text();
    reg.register(
        "help",
        "顯示可用指令列表",
        "/help",
        Arc::new(move |_sender, _args| {
            let text = help_text.clone();
            Box::pin(async move { Ok(text) })
        }),
    );
}

/// 從移除結果文字中提取 id（格式：「已移除XXX「id」」）
fn extract_removed_id(text: &str, _kind: &str) -> Option<String> {
    let start = text.find('「')? + '「'.len_utf8();
    let end = text[start..].find('」')? + start;
    Some(text[start..end].to_owned())
}
