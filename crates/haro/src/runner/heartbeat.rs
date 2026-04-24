use std::sync::Arc;

use anyhow::Result;
use tracing::{error, info};

use crate::api::{self, LlmProvider};
use crate::channel::Notifier;
use crate::{db, tool};

pub(crate) const HEARTBEAT_TRIAGE_PROMPT: &str = "\
You are a triage agent. Given the heartbeat instructions, current time, and scratchpad context, \
decide whether there is a task that needs to be executed RIGHT NOW.\n\
Consider time-sensitive tasks, scheduled actions, and pending items.\n\n\
Reply with ONLY \"yes\" (needs action now) or \"no\" (skip this cycle).";

/// 心跳任務設定
pub(crate) struct HeartbeatTask {
    pub pg: Arc<sqlx::PgPool>,
    pub worker: Arc<api::gemini::GeminiProvider>,
    pub llm: Arc<api::gemini::GeminiProvider>,
    pub agent_id: String,
    pub agent_prompt: String,
    pub heartbeat_prompt: String,
    pub interval_secs: u64,
    pub llm_model: String,
    pub notifiers: Vec<Arc<dyn Notifier>>,
    pub searxng_url: Option<String>,
    pub saachi_url: Option<String>,
    pub agent_path: Option<std::path::PathBuf>,
}

/// 心跳任務：定期喚醒 worker 判斷是否需要呼叫 LLM 執行任務
pub(crate) async fn run_heartbeat(task: HeartbeatTask) -> Result<()> {
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
        if let Some(ref url) = task.saachi_url {
            tools.register(tool::saachi::tool(url.clone()));
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
            "\n\n重要：你的直接回覆不會被任何人看到。所有需要回報的內容都必須使用 send_message 工具發送到頻道，否則使用者不會收到。"
        };

        let system = format!(
            "你是一個具有長期記憶的 AI 助手。\n\n\
             目前時間：{now}\n\n\
             {}\n\n\
             ## 備忘錄 (Scratchpad)\n{memo}\n\n\
             ## 指示\n\
             這是一個定期心跳喚醒。請根據以下心跳指示執行任務。\n\
             如果有重要的事情需要記住，請使用 save_memo 工具儲存到備忘錄中。\n\
             備忘錄只用於紀錄固定且明確的資訊（例如偏好、規則、待辦事項），不要紀錄曾經做過的事情。{notify_instruction}\n\n\
             ## 心跳指示\n{}",
            task.agent_prompt, task.heartbeat_prompt
        );

        let result = api::chat_with_tools(
            task.llm.as_ref(),
            &system,
            "心跳觸發，請執行任務。",
            &tools,
            None,
        )
        .await;

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
