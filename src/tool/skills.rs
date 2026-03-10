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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_skills_dir() -> TempDir {
        let dir = TempDir::new().unwrap();
        let skills = dir.path();
        fs::write(skills.join("coding.md"), "# Coding\nWrite clean code.").unwrap();
        fs::write(skills.join("translate.md"), "# Translation\nTranslate text.").unwrap();
        fs::write(skills.join("empty.md"), "").unwrap();
        fs::write(skills.join("not_a_skill.txt"), "ignored").unwrap();
        dir
    }

    #[test]
    fn list_entries_finds_md_files() {
        let dir = setup_skills_dir();
        let entries = list_skill_entries(dir.path());

        assert_eq!(entries.len(), 3);
        // 按名稱排序
        assert_eq!(entries[0].0, "coding");
        assert_eq!(entries[0].1, "Coding");
        assert_eq!(entries[1].0, "empty");
        assert_eq!(entries[1].1, "");
        assert_eq!(entries[2].0, "translate");
        assert_eq!(entries[2].1, "Translation");
    }

    #[test]
    fn list_entries_ignores_non_md() {
        let dir = setup_skills_dir();
        let entries = list_skill_entries(dir.path());
        let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
        assert!(!names.contains(&"not_a_skill"));
    }

    #[test]
    fn list_entries_nonexistent_dir() {
        let entries = list_skill_entries(Path::new("/nonexistent/path"));
        assert!(entries.is_empty());
    }

    #[test]
    fn list_entries_empty_dir() {
        let dir = TempDir::new().unwrap();
        let entries = list_skill_entries(dir.path());
        assert!(entries.is_empty());
    }

    #[test]
    fn heading_prefix_stripped() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.md"), "## Sub heading\nBody").unwrap();
        let entries = list_skill_entries(dir.path());
        assert_eq!(entries[0].1, "Sub heading");
    }

    #[tokio::test]
    async fn list_skills_tool_output() {
        let dir = setup_skills_dir();
        let tool_list = tools(dir.path().to_path_buf());
        let list_tool = &tool_list[0];
        assert_eq!(list_tool.definition().name, "list_skills");

        let result = list_tool.call(json!({})).await.unwrap();
        assert!(result.contains("coding — Coding"));
        assert!(result.contains("translate — Translation"));
    }

    #[tokio::test]
    async fn list_skills_tool_empty() {
        let dir = TempDir::new().unwrap();
        let tool_list = tools(dir.path().to_path_buf());
        let result = tool_list[0].call(json!({})).await.unwrap();
        assert!(result.contains("沒有可用"));
    }

    #[tokio::test]
    async fn get_skill_returns_content_and_path() {
        let dir = setup_skills_dir();
        let tool_list = tools(dir.path().to_path_buf());
        let get_tool = &tool_list[1];
        assert_eq!(get_tool.definition().name, "get_skill");

        let result = get_tool.call(json!({"name": "coding"})).await.unwrap();
        assert!(result.contains("## Skill: coding"));
        assert!(result.contains("base_path"));
        assert!(result.contains("Write clean code."));
    }

    #[tokio::test]
    async fn get_skill_not_found() {
        let dir = setup_skills_dir();
        let tool_list = tools(dir.path().to_path_buf());
        let result = tool_list[1].call(json!({"name": "nope"})).await.unwrap();
        assert!(result.contains("找不到"));
    }

    #[tokio::test]
    async fn get_skill_path_traversal_blocked() {
        let dir = setup_skills_dir();
        let tool_list = tools(dir.path().to_path_buf());
        let result = tool_list[1].call(json!({"name": "../etc/passwd"})).await.unwrap();
        assert!(result.contains("無效"));
    }

    #[tokio::test]
    async fn get_skill_empty_name() {
        let dir = setup_skills_dir();
        let tool_list = tools(dir.path().to_path_buf());
        let result = tool_list[1].call(json!({"name": ""})).await.unwrap();
        assert!(result.contains("未提供"));
    }
}
