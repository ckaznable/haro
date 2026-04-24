use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use chrono::{Local, Offset};
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
    pub saachi_url: Option<String>,
    pub skills_path: Option<std::path::PathBuf>,
}

/// 管理所有排程任務的執行池
pub(crate) struct RunningPool {
    res: Arc<SchedulerResources>,
    handles: std::sync::Mutex<HashMap<String, AbortHandle>>,
}

const TASK_SYNC_INTERVAL_SECS: u64 = 10;

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
        info!(
            agent_id = %self.res.agent_id,
            tasks_path = %tasks_path.display(),
            task_count = tasks_config.tasks.len(),
            "載入一次性任務設定"
        );
        for task in &tasks_config.tasks {
            self.spawn_task(task);
        }

        // 低頻同步：讓 add_task（非 /task 路徑）也能被 runtime 讀到
        self.spawn_task_sync_loop(tasks_path.to_path_buf());
    }

    /// 啟動單一 cron job（自帶內部迴圈）
    pub fn spawn_cron(self: &Arc<Self>, job: &CronJob) {
        let id = job.id.clone();
        let cron_expr = job.cron.clone();
        let prompt = job.prompt.clone();
        let executor_type = job.executor.clone();
        let pool = Arc::clone(self);
        let pool_key = cron_runtime_key(&id);
        let handle_key = pool_key.clone();

        // 先中止同名的舊 task（避免孤兒 leak）
        self.abort_cron(&id);

        let handle = tokio::spawn(async move {
            run_cron_loop(&pool.res, &id, &cron_expr, &prompt, &executor_type).await;
            pool.handles.lock().unwrap().remove(&handle_key);
        });

        self.handles
            .lock()
            .unwrap()
            .insert(pool_key, handle.abort_handle());
    }

    /// 啟動單一一次性任務
    pub fn spawn_task(self: &Arc<Self>, task: &ScheduledTask) {
        let id = task.id.clone();
        let run_at = task.run_at.clone();
        let prompt = task.prompt.clone();
        let executor_type = task.executor.clone();
        let pool = Arc::clone(self);
        let pool_key = task_runtime_key(&id);
        let handle_key = pool_key.clone();

        // 先中止同名的舊 task（避免孤兒 leak）
        self.abort_task(&id);

        info!(
            agent_id = %self.res.agent_id,
            task_id = %id,
            run_at = %run_at,
            executor = %executor_type,
            "註冊一次性任務 runtime"
        );

        let handle = tokio::spawn(async move {
            match run_once(&pool.res, &id, &run_at, &prompt, &executor_type).await {
                Ok(()) => {
                    // 僅在成功執行後才從 tasks.toml 刪除，避免失敗任務靜默消失
                    if let Err(e) = tool::task::remove_task(&pool.res.tasks_path, &id) {
                        error!(agent_id = %pool.res.agent_id, task_id = %id, "刪除已完成任務失敗: {e:#}");
                    }
                }
                Err(e) => {
                    error!(
                        agent_id = %pool.res.agent_id,
                        task_id = %id,
                        run_at = %run_at,
                        executor = %executor_type,
                        "一次性任務執行失敗，保留 tasks.toml 項目以便重試: {e:#}"
                    );
                }
            }
            pool.handles.lock().unwrap().remove(&handle_key);
        });

        self.handles
            .lock()
            .unwrap()
            .insert(pool_key, handle.abort_handle());
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

    pub fn abort_cron(&self, id: &str) -> bool {
        self.abort(&cron_runtime_key(id))
    }

    pub fn abort_task(&self, id: &str) -> bool {
        self.abort(&task_runtime_key(id))
    }

    /// 列出目前正在執行的事件 ID
    pub fn running_cron_ids(&self) -> Vec<String> {
        self.running_ids_with_prefix("cron:")
    }

    pub fn running_task_ids(&self) -> Vec<String> {
        self.running_ids_with_prefix("task:")
    }

    fn running_ids_with_prefix(&self, prefix: &str) -> Vec<String> {
        let guard = self.handles.lock().unwrap();
        guard
            .iter()
            .filter(|(id, h)| id.starts_with(prefix) && !h.is_finished())
            .map(|(id, _)| id.trim_start_matches(prefix).to_owned())
            .collect()
    }

    fn spawn_task_sync_loop(self: &Arc<Self>, tasks_path: std::path::PathBuf) {
        let pool = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker =
                tokio::time::interval(std::time::Duration::from_secs(TASK_SYNC_INTERVAL_SECS));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                ticker.tick().await;
                if let Err(e) = pool.sync_missing_tasks_from_disk(&tasks_path) {
                    error!(
                        agent_id = %pool.res.agent_id,
                        tasks_path = %tasks_path.display(),
                        "低頻同步一次性任務失敗: {e:#}"
                    );
                }
            }
        });
    }

    fn sync_missing_tasks_from_disk(self: &Arc<Self>, tasks_path: &std::path::Path) -> Result<()> {
        let config = tool::task::load_config(tasks_path)?;
        let running: HashSet<String> = self.running_task_ids().into_iter().collect();

        let mut spawned = 0usize;
        for task in &config.tasks {
            if running.contains(&task.id) {
                continue;
            }
            info!(
                agent_id = %self.res.agent_id,
                task_id = %task.id,
                run_at = %task.run_at,
                executor = %task.executor,
                "偵測到尚未載入 runtime 的一次性任務，補註冊"
            );
            self.spawn_task(task);
            spawned += 1;
        }

        if spawned > 0 {
            info!(
                agent_id = %self.res.agent_id,
                spawned,
                interval_secs = TASK_SYNC_INTERVAL_SECS,
                "一次性任務低頻同步完成"
            );
        }

        Ok(())
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
        if let Err(e) = execute_job(res, &format!("cron「{id}」"), prompt, executor_type).await {
            error!(agent_id = %res.agent_id, job_id = %id, executor = %executor_type, "cron 任務失敗: {e:#}");
        }
    }
}

/// 一次性任務：等到指定時間後執行一次
async fn run_once(
    res: &SchedulerResources,
    id: &str,
    run_at: &str,
    prompt: &str,
    executor_type: &Executor,
) -> Result<()> {
    let dt = parse_task_run_at(run_at).map_err(|e| anyhow!("task_id={id} 時間格式無效: {e}"))?;

    let now = Local::now();
    let wait = if dt > now {
        (dt - now).to_std().unwrap_or(std::time::Duration::ZERO)
    } else {
        std::time::Duration::ZERO
    };
    let overdue_secs = if dt <= now {
        (now - dt).num_seconds().max(0)
    } else {
        0
    };
    info!(
        agent_id = %res.agent_id,
        task_id = %id,
        run_at = %dt.format("%Y-%m-%d %H:%M:%S %:z"),
        now = %now.format("%Y-%m-%d %H:%M:%S %:z"),
        wait_secs = wait.as_secs(),
        overdue_secs,
        "一次性任務等待時間計算完成"
    );
    if !wait.is_zero() {
        tokio::time::sleep(wait).await;
    }

    info!(agent_id = %res.agent_id, task_id = %id, executor = %executor_type, "觸發一次性任務");
    execute_job(res, &format!("一次性任務「{id}」"), prompt, executor_type).await
}

/// 實際執行排程任務（cron / 一次性共用）
async fn execute_job(
    res: &SchedulerResources,
    event_label: &str,
    prompt: &str,
    executor_type: &Executor,
) -> Result<()> {
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
    if let Some(ref url) = res.saachi_url {
        tools.register(tool::saachi::tool(url.clone()));
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

    let r = api::chat_with_tools(executor, &system, &user_message, &tools, None)
        .await
        .map_err(|e| anyhow!("LLM 執行失敗: {e:#}"))?;

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
    Ok(())
}

fn cron_runtime_key(id: &str) -> String {
    format!("cron:{id}")
}

fn task_runtime_key(id: &str) -> String {
    format!("task:{id}")
}

fn parse_task_run_at(run_at: &str) -> Result<chrono::DateTime<Local>> {
    let dt = chrono::DateTime::parse_from_rfc3339(run_at).or_else(|_| {
        chrono::NaiveDateTime::parse_from_str(run_at, "%Y-%m-%d %H:%M")
            .or_else(|_| chrono::NaiveDateTime::parse_from_str(run_at, "%Y-%m-%dT%H:%M"))
            .or_else(|_| chrono::NaiveDateTime::parse_from_str(run_at, "%Y-%m-%d %H:%M:%S"))
            .or_else(|_| chrono::NaiveDateTime::parse_from_str(run_at, "%Y-%m-%dT%H:%M:%S"))
            .map(|naive| {
                let local_offset = Local::now().offset().fix();
                naive
                    .and_local_timezone(local_offset)
                    .single()
                    .ok_or_else(|| anyhow!("本地時間無法唯一解析"))
            })?
    })?;
    Ok(dt.with_timezone(&Local))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_keys_do_not_collide_between_cron_and_task() {
        assert_ne!(cron_runtime_key("same-id"), task_runtime_key("same-id"));
    }

    #[test]
    fn parse_task_run_at_rejects_invalid_datetime() {
        assert!(parse_task_run_at("not-a-datetime").is_err());
    }

    #[test]
    fn parse_task_run_at_accepts_legacy_local_format() {
        assert!(parse_task_run_at("2026-04-01 12:34").is_ok());
    }
}
