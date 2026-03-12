use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::str::FromStr;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{Tool, ToolDef};

/// Cron 任務設定檔（cron.toml）
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CronConfig {
    #[serde(default)]
    pub jobs: Vec<CronJob>,
}

/// 單一 Cron 任務
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CronJob {
    pub id: String,
    /// 標準 5 欄位 cron 表達式（分 時 日 月 週）
    pub cron: String,
    /// 執行時的提示詞
    pub prompt: String,
    /// 執行者：worker / llm / brain
    #[serde(default)]
    pub executor: Executor,
    /// 是否啟用
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Executor {
    Worker,
    #[default]
    Llm,
    Brain,
}

impl std::fmt::Display for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Worker => write!(f, "worker"),
            Self::Llm => write!(f, "llm"),
            Self::Brain => write!(f, "brain"),
        }
    }
}

/// 載入 cron.toml（不存在則回傳空設定）
pub fn load_config(path: &std::path::Path) -> Result<CronConfig> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(toml::from_str(&s)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(CronConfig::default()),
        Err(e) => Err(e.into()),
    }
}

/// 寫入 cron.toml
pub fn save_config(path: &std::path::Path, config: &CronConfig) -> Result<()> {
    let s = toml::to_string_pretty(config)?;
    std::fs::write(path, s)?;
    Ok(())
}

/// 驗證 cron 表達式是否合法
fn validate_cron(expr: &str) -> Result<(), String> {
    // cron crate 需要 7 欄位（秒 分 時 日 月 週 年），我們接受 5 欄位並補上秒和年
    let full = format!("0 {expr} *");
    cron::Schedule::from_str(&full).map(|_| ()).map_err(|e| e.to_string())
}

/// 格式化任務列表為可讀文字
fn format_jobs(jobs: &[CronJob]) -> String {
    if jobs.is_empty() {
        return "目前沒有排程任務。".into();
    }
    jobs.iter()
        .map(|j| {
            let status = if j.enabled { "啟用" } else { "停用" };
            let prompt_preview: String = j.prompt.chars().take(60).collect();
            let ellipsis = if j.prompt.chars().count() > 60 { "…" } else { "" };
            format!(
                "• {} [{}] ({}, {})\n  cron: {}\n  prompt: {}{ellipsis}",
                j.id, status, j.executor, j.cron, j.cron, prompt_preview
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// 回傳 cron 任務管理工具
pub fn tools(agent_path: PathBuf) -> Vec<Box<dyn Tool>> {
    let cron_path = agent_path.join("cron.toml");
    vec![
        Box::new(ListCronTool {
            path: cron_path.clone(),
        }),
        Box::new(AddCronTool {
            path: cron_path.clone(),
        }),
        Box::new(EditCronTool {
            path: cron_path.clone(),
        }),
        Box::new(RemoveCronTool { path: cron_path }),
    ]
}

// ── list_cron ──

struct ListCronTool {
    path: PathBuf,
}

impl Tool for ListCronTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "list_cron".into(),
            description: "List all scheduled cron jobs with their id, cron expression, prompt, executor, and status."
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
            Ok(format_jobs(&config.jobs))
        })
    }
}

// ── add_cron ──

struct AddCronTool {
    path: PathBuf,
}

impl Tool for AddCronTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "add_cron".into(),
            description: "Add a new scheduled cron job. The cron expression uses standard 5-field format \
                (minute hour day-of-month month day-of-week). The executor specifies which model runs \
                the prompt: 'worker' (fast/cheap), 'llm' (main chat model), or 'brain' (most capable)."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique identifier for the job (slug, e.g. 'morning-brief')"
                    },
                    "cron": {
                        "type": "string",
                        "description": "Cron expression (5-field: minute hour day month weekday)"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt/instruction to execute on schedule"
                    },
                    "executor": {
                        "type": "string",
                        "enum": ["worker", "llm", "brain"],
                        "description": "Which model executes the prompt (default: llm)"
                    }
                },
                "required": ["id", "cron", "prompt"]
            }),
        }
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let id = args.get("id").and_then(|v| v.as_str()).unwrap_or_default();
            let cron_expr = args.get("cron").and_then(|v| v.as_str()).unwrap_or_default();
            let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or_default();
            let executor: Executor = args
                .get("executor")
                .and_then(|v| v.as_str())
                .and_then(|s| serde_json::from_value(json!(s)).ok())
                .unwrap_or_default();

            if id.is_empty() || cron_expr.is_empty() || prompt.is_empty() {
                return Ok("錯誤：id、cron、prompt 皆為必填。".into());
            }

            if let Err(e) = validate_cron(cron_expr) {
                return Ok(format!("cron 表達式無效: {e}"));
            }

            let mut config = load_config(&self.path)?;

            if config.jobs.iter().any(|j| j.id == id) {
                return Ok(format!("錯誤：已存在 id 為「{id}」的任務。"));
            }

            config.jobs.push(CronJob {
                id: id.to_owned(),
                cron: cron_expr.to_owned(),
                prompt: prompt.to_owned(),
                executor,
                enabled: true,
            });

            save_config(&self.path, &config)?;
            Ok(format!("已新增排程任務「{id}」"))
        })
    }
}

// ── edit_cron ──

struct EditCronTool {
    path: PathBuf,
}

impl Tool for EditCronTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "edit_cron".into(),
            description: "Edit an existing cron job. Only the specified fields will be updated."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The job id to edit"
                    },
                    "cron": {
                        "type": "string",
                        "description": "New cron expression (optional)"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "New prompt text (optional)"
                    },
                    "executor": {
                        "type": "string",
                        "enum": ["worker", "llm", "brain"],
                        "description": "New executor (optional)"
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "Enable or disable the job (optional)"
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
            let Some(job) = config.jobs.iter_mut().find(|j| j.id == id) else {
                return Ok(format!("找不到 id 為「{id}」的任務。"));
            };

            if let Some(cron_expr) = args.get("cron").and_then(|v| v.as_str()) {
                if let Err(e) = validate_cron(cron_expr) {
                    return Ok(format!("cron 表達式無效: {e}"));
                }
                job.cron = cron_expr.to_owned();
            }
            if let Some(prompt) = args.get("prompt").and_then(|v| v.as_str()) {
                job.prompt = prompt.to_owned();
            }
            if let Some(executor) = args
                .get("executor")
                .and_then(|v| v.as_str())
                .and_then(|s| serde_json::from_value(json!(s)).ok())
            {
                job.executor = executor;
            }
            if let Some(enabled) = args.get("enabled").and_then(|v| v.as_bool()) {
                job.enabled = enabled;
            }

            save_config(&self.path, &config)?;
            Ok(format!("已更新排程任務「{id}」"))
        })
    }
}

// ── remove_cron ──

struct RemoveCronTool {
    path: PathBuf,
}

impl Tool for RemoveCronTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "remove_cron".into(),
            description: "Remove a cron job by its id.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The job id to remove"
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
            let before = config.jobs.len();
            config.jobs.retain(|j| j.id != id);

            if config.jobs.len() == before {
                return Ok(format!("找不到 id 為「{id}」的任務。"));
            }

            save_config(&self.path, &config)?;
            Ok(format!("已移除排程任務「{id}」"))
        })
    }
}

/// 處理 /cron slash command 的子指令
pub fn handle_slash_command(cron_path: &std::path::Path, args: &str) -> Result<String> {
    let args = args.trim();

    if args.is_empty() || args == "list" {
        let config = load_config(cron_path)?;
        return Ok(format_jobs(&config.jobs));
    }

    if let Some(id) = args.strip_prefix("remove ").or_else(|| args.strip_prefix("rm ")) {
        let id = id.trim();
        let mut config = load_config(cron_path)?;
        let before = config.jobs.len();
        config.jobs.retain(|j| j.id != id);
        if config.jobs.len() == before {
            return Ok(format!("找不到 id 為「{id}」的任務。"));
        }
        save_config(cron_path, &config)?;
        return Ok(format!("已移除排程任務「{id}」"));
    }

    if let Some(rest) = args.strip_prefix("enable ") {
        let id = rest.trim();
        return toggle_job(cron_path, id, true);
    }

    if let Some(rest) = args.strip_prefix("disable ") {
        let id = rest.trim();
        return toggle_job(cron_path, id, false);
    }

    // add <id> <cron_expr(5 fields)> <prompt...>
    if let Some(rest) = args.strip_prefix("add ") {
        return add_job_from_args(cron_path, rest.trim());
    }

    Ok("用法:\n\
        /cron — 列出所有排程\n\
        /cron add <id> <分 時 日 月 週> <提示詞>\n\
        /cron remove <id>\n\
        /cron enable <id>\n\
        /cron disable <id>"
        .into())
}

fn toggle_job(cron_path: &std::path::Path, id: &str, enabled: bool) -> Result<String> {
    let mut config = load_config(cron_path)?;
    let Some(job) = config.jobs.iter_mut().find(|j| j.id == id) else {
        return Ok(format!("找不到 id 為「{id}」的任務。"));
    };
    job.enabled = enabled;
    save_config(cron_path, &config)?;
    let status = if enabled { "啟用" } else { "停用" };
    Ok(format!("排程任務「{id}」已{status}"))
}

fn add_job_from_args(cron_path: &std::path::Path, rest: &str) -> Result<String> {
    // 格式: <id> <min> <hour> <dom> <mon> <dow> <prompt...>
    let parts: Vec<&str> = rest.splitn(7, ' ').collect();
    if parts.len() < 7 {
        return Ok("用法: /cron add <id> <分> <時> <日> <月> <週> <提示詞>".into());
    }

    let id = parts[0];
    let cron_expr = format!("{} {} {} {} {}", parts[1], parts[2], parts[3], parts[4], parts[5]);
    let prompt = parts[6];

    if let Err(e) = validate_cron(&cron_expr) {
        return Ok(format!("cron 表達式無效: {e}"));
    }

    let mut config = load_config(cron_path)?;
    if config.jobs.iter().any(|j| j.id == id) {
        return Ok(format!("錯誤：已存在 id 為「{id}」的任務。"));
    }

    config.jobs.push(CronJob {
        id: id.to_owned(),
        cron: cron_expr,
        prompt: prompt.to_owned(),
        executor: Executor::default(),
        enabled: true,
    });

    save_config(cron_path, &config)?;
    Ok(format!("已新增排程任務「{id}」"))
}
