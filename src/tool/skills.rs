use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use anyhow::Result;
use serde_json::json;

use super::{Tool, ToolDef};

/// 掃描 skills 目錄，回傳 (name, 第一行描述) 列表
pub fn list_skill_entries(skills_path: &Path) -> Vec<(String, String)> {
    let Ok(entries) = std::fs::read_dir(skills_path) else {
        return vec![];
    };

    let mut skills: Vec<(String, String)> = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().is_some_and(|ext| ext == "md")
                && e.file_type().is_ok_and(|ft| ft.is_file())
        })
        .filter_map(|e| {
            let name = e.path().file_stem()?.to_str()?.to_owned();
            let content = std::fs::read_to_string(e.path()).ok()?;
            let first_line = content.lines().next().unwrap_or("").trim().to_owned();
            // 去掉 markdown heading 前綴
            let desc = first_line
                .trim_start_matches('#')
                .trim()
                .to_owned();
            Some((name, desc))
        })
        .collect();

    skills.sort_by(|a, b| a.0.cmp(&b.0));
    skills
}

/// 建立 skills 工具（list_skills + get_skill）
pub fn tools(skills_path: PathBuf) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ListSkillsTool {
            skills_path: skills_path.clone(),
        }),
        Box::new(GetSkillTool { skills_path }),
    ]
}

// ── list_skills ──

struct ListSkillsTool {
    skills_path: PathBuf,
}

impl Tool for ListSkillsTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "list_skills".into(),
            description: "List all available skills for this agent. \
                Returns skill names and their descriptions."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    fn call(
        &self,
        _args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async {
            let entries = list_skill_entries(&self.skills_path);
            if entries.is_empty() {
                return Ok("目前沒有可用的 skills。".into());
            }

            let list: Vec<String> = entries
                .iter()
                .map(|(name, desc)| {
                    if desc.is_empty() {
                        name.clone()
                    } else {
                        format!("{name} — {desc}")
                    }
                })
                .collect();

            Ok(list.join("\n"))
        })
    }
}

// ── get_skill ──

struct GetSkillTool {
    skills_path: PathBuf,
}

impl Tool for GetSkillTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "get_skill".into(),
            description: "Load a specific skill by name. \
                Returns the full skill content and its base_path for context."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name (without .md extension)"
                    }
                },
                "required": ["name"]
            }),
        }
    }

    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let name = args
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            if name.is_empty() {
                return Ok("錯誤：未提供 skill 名稱".into());
            }

            // 防止路徑穿越
            if name.contains('/') || name.contains('\\') || name.contains("..") {
                return Ok("錯誤：無效的 skill 名稱".into());
            }

            let file_path = self.skills_path.join(format!("{name}.md"));
            let content = match std::fs::read_to_string(&file_path) {
                Ok(c) => c,
                Err(_) => return Ok(format!("錯誤：找不到 skill「{name}」")),
            };

            let base_path = self.skills_path.display();
            Ok(format!(
                "## Skill: {name}\n\
                 **base_path**: {base_path}\n\n\
                 ---\n\n\
                 {content}"
            ))
        })
    }
}
