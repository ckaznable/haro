use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use chrono::Local;
use tokio::task::AbortHandle;
use tracing::{error, info, warn};

use crate::api;
use crate::channel::Notifier;
use crate::{db, tool};
use tool::cron::{CronJob, Executor};
use tool::task::ScheduledTask;

/// 執行排程任務所需的共用資源
pub(crate) struct SchedulerResources {
    pub pg: Arc<sqlx::PgPool>,
    pub worker: Arc<api::gemini::GeminiProvider>,
    pub llm: Arc<api::gemini::GeminiProvider>,
    pub brain: Arc<api::gemini::GeminiProvider>,
    pub agent_id: String,
    pub agent_prompt: String,
    pub tasks_path: std::path::PathBuf,
    pub llm_model: String,
    pub brain_model: String,
    pub worker_model: String,
    pub notifiers: Vec<Arc<dyn Notifier>>,
    pub searxng_url: Option<String>,
    pub skills_path: Option<std::path::PathBuf>,
}

/// 管理所有排程任務的執行池
pub(crate) struct RunningPool {
    res: Arc<SchedulerResources>,
    handles: std::sync::Mutex<HashMap<String, AbortHandle>>,
}

impl RunningPool {
    pub fn new(res: SchedulerResources) -> Arc<Self> {
        Arc::new(Self {
            res: Arc::new(res),
            handles: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// 啟動時載入所有 cron jobs 和 scheduled tasks
    pub fn spawn_all(self: &Arc<Self>, cron_path: &std::path::Path, tasks_path: &std::path::Path) {
        // Cron jobs
        let config = tool::cron::load_config(cron_path).unwrap_or_default();
        for job in config.jobs.iter().filter(|j| j.enabled) {
            self.spawn_cron(job);
        }

        // One-time tasks
        let tasks_config = tool::task::load_config(tasks_path).unwrap_or_default();
        for task in &tasks_config.tasks {
            self.spawn_task(task);
        }
    }

    /// 啟動單一 cron job（自帶內部迴圈）
    pub fn spawn_cron(self: &Arc<Self>, job: &CronJob) {
        let id = job.id.clone();
        let cron_expr = job.cron.clone();
        let prompt = job.prompt.clone();
        let executor_type = job.executor.clone();
        let pool = Arc::clone(self);
        let pool_key = id.clone();

        // 先中止同名的舊 task（避免孤兒 leak）
        self.abort(&id);

        let handle = tokio::spawn(async move {
            run_cron_loop(&pool.res, &pool_key, &cron_expr, &prompt, &executor_type).await;
            pool.handles.lock().unwrap().remove(&pool_key);
        });

        self.handles
            .lock()
            .unwrap()
            .insert(id, handle.abort_handle());
    }

    /// 啟動單一一次性任務
    pub fn spawn_task(self: &Arc<Self>, task: &ScheduledTask) {
        let id = task.id.clone();
        let run_at = task.run_at.clone();
        let prompt = task.prompt.clone();
        let executor_type = task.executor.clone();
        let pool = Arc::clone(self);
        let pool_key = id.clone();

        // 先中止同名的舊 task（避免孤兒 leak）
        self.abort(&id);

        let handle = tokio::spawn(async move {
            run_once(&pool.res, &pool_key, &run_at, &prompt, &executor_type).await;
            // 完成後從 tasks.toml 刪除
            if let Err(e) = tool::task::remove_task(&pool.res.tasks_path, &pool_key) {
                error!(agent_id = %pool.res.agent_id, task_id = %pool_key, "刪除已完成任務失敗: {e:#}");
            }
            pool.handles.lock().unwrap().remove(&pool_key);
        });

        self.handles
            .lock()
            .unwrap()
            .insert(id, handle.abort_handle());
    }

    /// 中止指定事件
    pub fn abort(&self, id: &str) -> bool {
        if let Some(handle) = self.handles.lock().unwrap().remove(id) {
            handle.abort();
            true
        } else {
            false
        }
    }

    /// 列出目前正在執行的事件 ID
    pub fn running_ids(&self) -> Vec<String> {
        let guard = self.handles.lock().unwrap();
        guard
            .iter()
            .filter(|(_, h)| !h.is_finished())
            .map(|(id, _)| id.clone())
            .collect()
    }
}

/// 單一 cron job 的內部迴圈
async fn run_cron_loop(
    res: &SchedulerResources,
    id: &str,
    cron_expr: &str,
    prompt: &str,
    executor_type: &Executor,
) {
    let full_expr = format!("0 {cron_expr} *");
    let schedule = match cron::Schedule::from_str(&full_expr) {
        Ok(s) => s,
        Err(e) => {
            error!(agent_id = %res.agent_id, job_id = %id, "cron 表達式無效: {e}");
            return;
        }
    };

    loop {
        let now = Local::now();
        let Some(next) = schedule.upcoming(Local).next() else {
            warn!(agent_id = %res.agent_id, job_id = %id, "無法計算下次觸發時間");
            return;
        };

        // 保證至少等 1 秒，避免邊界情況 busy loop
        let wait = (next - now)
            .to_std()
            .unwrap_or(std::time::Duration::from_secs(1))
            .max(std::time::Duration::from_secs(1));
        info!(
            agent_id = %res.agent_id,
            job_id = %id,
            next_time = %next.format("%H:%M:%S"),
            wait_secs = wait.as_secs(),
            "cron 等待下次觸發"
        );
        tokio::time::sleep(wait).await;

        info!(agent_id = %res.agent_id, job_id = %id, executor = %executor_type, "觸發 cron 任務");
        execute_job(res, &format!("cron「{id}」"), prompt, executor_type).await;
    }
}

/// 一次性任務：等到指定時間後執行一次
async fn run_once(
    res: &SchedulerResources,
    id: &str,
    run_at: &str,
    prompt: &str,
    executor_type: &Executor,
) {
    let dt = match chrono::DateTime::parse_from_rfc3339(run_at) {
        Ok(dt) => dt.with_timezone(&Local),
        Err(e) => {
            error!(agent_id = %res.agent_id, task_id = %id, "時間格式無效: {e}");
            return;
        }
    };

    let now = Local::now();
    if dt > now {
        let wait = (dt - now).to_std().unwrap_or(std::time::Duration::ZERO);
        info!(
            agent_id = %res.agent_id,
            task_id = %id,
            run_at = %dt.format("%Y-%m-%d %H:%M:%S"),
            wait_secs = wait.as_secs(),
            "一次性任務等待觸發"
        );
        tokio::time::sleep(wait).await;
    }

    info!(agent_id = %res.agent_id, task_id = %id, executor = %executor_type, "觸發一次性任務");
    execute_job(res, &format!("一次性任務「{id}」"), prompt, executor_type).await;
}

/// 實際執行排程任務（cron / 一次性共用）
async fn execute_job(
    res: &SchedulerResources,
    event_label: &str,
    prompt: &str,
    executor_type: &Executor,
) {
    let (executor, model_name): (&api::gemini::GeminiProvider, &str) = match executor_type {
        Executor::Worker => (res.worker.as_ref(), &res.worker_model),
        Executor::Llm => (res.llm.as_ref(), &res.llm_model),
        Executor::Brain => (res.brain.as_ref(), &res.brain_model),
    };

    let mut tools = tool::ToolRegistry::new();
    for t in tool::scratchpad::tools(Arc::clone(&res.pg), res.agent_id.clone()) {
        tools.register(t);
    }
    tools.register(tool::fetch::tool());
    if executor.grounding() {
        let executor_arc: Arc<api::gemini::GeminiProvider> = match executor_type {
            Executor::Worker => Arc::clone(&res.worker),
            Executor::Llm => Arc::clone(&res.llm),
            Executor::Brain => Arc::clone(&res.brain),
        };
        tools.register(tool::search::tool(executor_arc));
    }
    if let Some(ref url) = res.searxng_url {
        tools.register(tool::searxng::tool(url.clone()));
    }
    if let Some(ref sp) = res.skills_path {
        for t in tool::skills::tools(sp.clone()) {
            tools.register(t);
        }
    }

    let memo = db::postgres::get_scratchpad(&res.pg, &res.agent_id)
        .await
        .unwrap_or(None)
        .unwrap_or_default();

    let now = Local::now().format("%Y-%m-%d %H:%M:%S %:z");
    let notify_instruction = if res.notifiers.is_empty() {
        ""
    } else {
        "\n\n重要：你的最終回覆會由系統自動推送到頻道，請直接輸出要給使用者看的內容。"
    };

    let system = format!(
        "你是一個 AI 助手，正在執行排程任務。\n\n\
         目前時間：{now}\n\n\
         {}\n\n\
         ## 備忘錄 (Scratchpad)\n{memo}\n\n\
         ## 指示\n\
         這是一個排程任務，請根據以下提示詞執行。\n\
         請直接輸出要回報給使用者的最終訊息內容，不需要呼叫任何發送工具。\n\
         如果需要搜尋最新資訊或外部資料，請務必使用可用的搜尋工具，不要憑空猜測。\n\
         如果有重要的事情需要記住，請使用 save_memo 工具儲存。{notify_instruction}",
        res.agent_prompt
    );

    let user_message = format!("{event_label}觸發，請執行：\n\n{prompt}");

    let result = api::chat_with_tools(executor, &system, &user_message, &tools, None).await;

    match result {
        Ok(r) => {
            let _ = db::postgres::insert_token_usage(
                &res.pg,
                &res.agent_id,
                model_name,
                r.input_tokens,
                r.output_tokens,
            )
            .await;

            if !res.notifiers.is_empty() {
                let message = r.text.trim();
                if message.is_empty() {
                    warn!(agent_id = %res.agent_id, event = %event_label, "排程任務回覆為空，略過通知");
                } else {
                    let mut sent = 0usize;
                    let mut failed = 0usize;
                    for notifier in &res.notifiers {
                        match notifier.send(message).await {
                            Ok(()) => sent += 1,
                            Err(e) => {
                                failed += 1;
                                error!(agent_id = %res.agent_id, event = %event_label, "推送排程結果失敗: {e:#}");
                            }
                        }
                    }
                    info!(
                        agent_id = %res.agent_id,
                        event = %event_label,
                        sent,
                        failed,
                        "排程結果已推送到 notifier"
                    );
                }
            }
            info!(agent_id = %res.agent_id, event = %event_label, "排程任務完成");
        }
        Err(e) => {
            error!(agent_id = %res.agent_id, event = %event_label, "排程任務失敗: {e:#}");
        }
    }
}
