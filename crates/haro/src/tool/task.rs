use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use anyhow::Result;
use chrono::Offset;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::cron::Executor;
use super::{Tool, ToolDef};

/// 一次性排程任務設定檔（tasks.toml）
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct TasksConfig {
    #[serde(default)]
    pub tasks: Vec<ScheduledTask>,
}

/// 一次性排程任務
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduledTask {
    pub id: String,
    /// 執行時間（ISO 8601，例如 "2026-03-12T14:00:00+08:00"）
    pub run_at: String,
    /// 執行時的提示詞
    pub prompt: String,
    /// 執行者
    #[serde(default)]
    pub executor: Executor,
}

/// 載入 tasks.toml（不存在則回傳空設定）
pub fn load_config(path: &std::path::Path) -> Result<TasksConfig> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(toml::from_str(&s)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(TasksConfig::default()),
        Err(e) => Err(e.into()),
    }
}

/// 寫入 tasks.toml
fn save_config(path: &std::path::Path, config: &TasksConfig) -> Result<()> {
    let s = toml::to_string_pretty(config)?;
    std::fs::write(path, s)?;
    Ok(())
}

/// 驗證 datetime 是否合法，回傳解析後的 DateTime
fn validate_datetime(s: &str) -> Result<chrono::DateTime<chrono::FixedOffset>, String> {
    chrono::DateTime::parse_from_rfc3339(s)
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M")
                .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M"))
                .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
                .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S"))
                .map(|naive| {
                    let local_offset = chrono::Local::now().offset().fix();
                    naive.and_local_timezone(local_offset).unwrap().fixed_offset()
                })
                .map_err(|e: chrono::ParseError| e.to_string())
        })
        .map_err(|e| format!("時間格式無效: {e}。支援 ISO 8601 或 YYYY-MM-DD HH:MM 格式"))
}

/// 格式化任務列表為可讀文字
fn format_tasks(tasks: &[ScheduledTask]) -> String {
    if tasks.is_empty() {
        return "目前沒有排定的一次性任務。".into();
    }
    tasks
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let prompt_preview: String = t.prompt.chars().take(60).collect();
            let ellipsis = if t.prompt.chars().count() > 60 { "…" } else { "" };
            format!(
                "#{} {} ({}, {})\n  時間: {}\n  prompt: {}{ellipsis}",
                i + 1, t.id, t.executor, t.run_at, t.run_at, prompt_preview
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// 回傳一次性任務管理工具
pub fn tools(agent_path: PathBuf) -> Vec<Box<dyn Tool>> {
    let tasks_path = agent_path.join("tasks.toml");
    vec![
        Box::new(ListTasksTool {
            path: tasks_path.clone(),
        }),
        Box::new(AddTaskTool {
            path: tasks_path.clone(),
        }),
        Box::new(EditTaskTool {
            path: tasks_path.clone(),
        }),
        Box::new(RemoveTaskTool { path: tasks_path }),
    ]
}

// ── list_tasks ──

struct ListTasksTool {
    path: PathBuf,
}

impl Tool for ListTasksTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "list_tasks".into(),
            description: "List all pending one-time scheduled tasks with their id, run_at time, prompt, and executor."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    fn call(
        &self,
        _args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let config = load_config(&self.path)?;
            Ok(format_tasks(&config.tasks))
        })
    }
}

// ── add_task ──

struct AddTaskTool {
    path: PathBuf,
}

impl Tool for AddTaskTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "add_task".into(),
            description: "Add a one-time scheduled task. It runs once at the specified time and is \
                then automatically removed. Time can be ISO 8601 (e.g. '2026-03-12T14:00:00+08:00') \
                or 'YYYY-MM-DD HH:MM' in local timezone."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique identifier for the task (slug, e.g. 'remind-meeting')"
                    },
                    "run_at": {
                        "type": "string",
                        "description": "When to execute (ISO 8601 or 'YYYY-MM-DD HH:MM')"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt/instruction to execute at the scheduled time"
                    },
                    "executor": {
                        "type": "string",
                        "enum": ["worker", "llm", "brain"],
                        "description": "Which model executes the prompt (default: llm)"
                    }
                },
                "required": ["id", "run_at", "prompt"]
            }),
        }
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let id = args.get("id").and_then(|v| v.as_str()).unwrap_or_default();
            let run_at = args.get("run_at").and_then(|v| v.as_str()).unwrap_or_default();
            let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or_default();
            let executor: Executor = args
                .get("executor")
                .and_then(|v| v.as_str())
                .and_then(|s| serde_json::from_value(json!(s)).ok())
                .unwrap_or_default();

            if id.is_empty() || run_at.is_empty() || prompt.is_empty() {
                return Ok("錯誤：id、run_at、prompt 皆為必填。".into());
            }

            let dt = match validate_datetime(run_at) {
                Ok(dt) => dt,
                Err(e) => return Ok(e),
            };

            if dt <= chrono::Local::now().fixed_offset() {
                return Ok("錯誤：排定時間必須是未來的時間。".into());
            }

            let mut config = load_config(&self.path)?;

            if config.tasks.iter().any(|t| t.id == id) {
                return Ok(format!("錯誤：已存在 id 為「{id}」的任務。"));
            }

            let run_at_str = dt.to_rfc3339();

            config.tasks.push(ScheduledTask {
                id: id.to_owned(),
                run_at: run_at_str.clone(),
                prompt: prompt.to_owned(),
                executor,
            });

            save_config(&self.path, &config)?;
            Ok(format!("已新增一次性任務「{id}」，將於 {run_at_str} 執行"))
        })
    }
}

// ── edit_task ──

struct EditTaskTool {
    path: PathBuf,
}

impl Tool for EditTaskTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "edit_task".into(),
            description: "Edit an existing one-time scheduled task. Only the specified fields will be updated."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The task id to edit"
                    },
                    "run_at": {
                        "type": "string",
                        "description": "New execution time (optional)"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "New prompt text (optional)"
                    },
                    "executor": {
                        "type": "string",
                        "enum": ["worker", "llm", "brain"],
                        "description": "New executor (optional)"
                    }
                },
                "required": ["id"]
            }),
        }
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let id = args.get("id").and_then(|v| v.as_str()).unwrap_or_default();
            if id.is_empty() {
                return Ok("錯誤：id 為必填。".into());
            }

            let mut config = load_config(&self.path)?;
            let Some(task) = config.tasks.iter_mut().find(|t| t.id == id) else {
                return Ok(format!("找不到 id 為「{id}」的任務。"));
            };

            if let Some(run_at) = args.get("run_at").and_then(|v| v.as_str()) {
                let dt = match validate_datetime(run_at) {
                    Ok(dt) => dt,
                    Err(e) => return Ok(e),
                };
                task.run_at = dt.to_rfc3339();
            }
            if let Some(prompt) = args.get("prompt").and_then(|v| v.as_str()) {
                task.prompt = prompt.to_owned();
            }
            if let Some(executor) = args
                .get("executor")
                .and_then(|v| v.as_str())
                .and_then(|s| serde_json::from_value(json!(s)).ok())
            {
                task.executor = executor;
            }

            save_config(&self.path, &config)?;
            Ok(format!("已更新一次性任務「{id}」"))
        })
    }
}

// ── remove_task ──

struct RemoveTaskTool {
    path: PathBuf,
}

impl Tool for RemoveTaskTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "remove_task".into(),
            description: "Remove a one-time scheduled task by its id.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The task id to remove"
                    }
                },
                "required": ["id"]
            }),
        }
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let id = args.get("id").and_then(|v| v.as_str()).unwrap_or_default();
            if id.is_empty() {
                return Ok("錯誤：id 為必填。".into());
            }

            let mut config = load_config(&self.path)?;
            let before = config.tasks.len();
            config.tasks.retain(|t| t.id != id);

            if config.tasks.len() == before {
                return Ok(format!("找不到 id 為「{id}」的任務。"));
            }

            save_config(&self.path, &config)?;
            Ok(format!("已移除一次性任務「{id}」"))
        })
    }
}

/// 處理 /task slash command 的子指令
pub fn handle_slash_command(tasks_path: &std::path::Path, args: &str) -> Result<String> {
    let args = args.trim();

    if args.is_empty() || args == "list" {
        let config = load_config(tasks_path)?;
        return Ok(format_tasks(&config.tasks));
    }

    if let Some(id_or_num) = args.strip_prefix("remove ").or_else(|| args.strip_prefix("rm ")) {
        let id_or_num = id_or_num.trim();
        let mut config = load_config(tasks_path)?;

        // 支援編號刪除（#N 或純數字）
        let id = if let Some(n) = id_or_num.strip_prefix('#').and_then(|s| s.parse::<usize>().ok())
            .or_else(|| id_or_num.parse::<usize>().ok())
        {
            if n == 0 || n > config.tasks.len() {
                return Ok(format!("編號 #{n} 不存在，共 {} 筆任務。", config.tasks.len()));
            }
            config.tasks[n - 1].id.clone()
        } else {
            id_or_num.to_owned()
        };

        let before = config.tasks.len();
        config.tasks.retain(|t| t.id != id);
        if config.tasks.len() == before {
            return Ok(format!("找不到 id 為「{id}」的任務。"));
        }
        save_config(tasks_path, &config)?;
        return Ok(format!("已移除一次性任務「{id}」"));
    }

    // add <id> <datetime> <prompt...>
    if let Some(rest) = args.strip_prefix("add ") {
        return add_task_from_args(tasks_path, rest.trim());
    }

    Ok("用法:\n\
        /task — 列出所有待執行任務\n\
        /task add <id> <時間> <提示詞>\n\
        /task remove <編號|id>"
        .into())
}

fn add_task_from_args(tasks_path: &std::path::Path, rest: &str) -> Result<String> {
    let parts: Vec<&str> = rest.splitn(2, ' ').collect();
    if parts.len() < 2 {
        return Ok("用法: /task add <id> <時間> <提示詞>".into());
    }

    let id = parts[0];
    let rest = parts[1];

    let (run_at_str, prompt) = match find_datetime_split(rest) {
        Some((dt, p)) => (dt, p),
        None => return Ok("時間格式無法辨識。支援 YYYY-MM-DD HH:MM 或 ISO 8601 格式".into()),
    };

    let dt = match validate_datetime(&run_at_str) {
        Ok(dt) => dt,
        Err(e) => return Ok(e),
    };

    if dt <= chrono::Local::now().fixed_offset() {
        return Ok("錯誤：排定時間必須是未來的時間。".into());
    }

    let mut config = load_config(tasks_path)?;
    if config.tasks.iter().any(|t| t.id == id) {
        return Ok(format!("錯誤：已存在 id 為「{id}」的任務。"));
    }

    let run_at_rfc = dt.to_rfc3339();
    config.tasks.push(ScheduledTask {
        id: id.to_owned(),
        run_at: run_at_rfc.clone(),
        prompt,
        executor: Executor::default(),
    });

    save_config(tasks_path, &config)?;
    Ok(format!("已新增一次性任務「{id}」，將於 {run_at_rfc} 執行"))
}

/// 嘗試從字串前面找到可解析的日期時間，回傳 (datetime_str, remaining_prompt)
fn find_datetime_split(s: &str) -> Option<(String, String)> {
    // 嘗試固定長度前綴
    for len in [19, 16] {
        if s.len() >= len {
            if validate_datetime(&s[..len]).is_ok() {
                let rest = s[len..].trim();
                if !rest.is_empty() {
                    return Some((s[..len].to_owned(), rest.to_owned()));
                }
            }
        }
    }

    // 嘗試用空格分割
    let parts: Vec<&str> = s.splitn(4, ' ').collect();

    // 前 2 個欄位 (如 "2026-03-12 14:00")
    if parts.len() >= 3 {
        let candidate = format!("{} {}", parts[0], parts[1]);
        if validate_datetime(&candidate).is_ok() {
            let rest: String = parts[2..].join(" ");
            if !rest.is_empty() {
                return Some((candidate, rest));
            }
        }
    }

    // 前 1 個欄位 (如 "2026-03-12T14:00:00+08:00")
    if parts.len() >= 2 {
        let candidate = parts[0];
        if validate_datetime(candidate).is_ok() {
            let rest: String = parts[1..].join(" ");
            if !rest.is_empty() {
                return Some((candidate.to_owned(), rest));
            }
        }
    }

    None
}

/// 刪除已完成的一次性任務
pub fn remove_task(tasks_path: &std::path::Path, id: &str) -> Result<()> {
    let mut config = load_config(tasks_path)?;
    config.tasks.retain(|t| t.id != id);
    save_config(tasks_path, &config)?;
    Ok(())
}
