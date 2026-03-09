pub mod fetch;
pub mod scratchpad;
pub mod skills;

use std::future::Future;
use std::pin::Pin;

use anyhow::Result;

/// 工具定義（傳給 LLM 的 function declaration）
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// LLM 可呼叫的工具（object-safe）
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDef;
    fn call(
        &self,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>>;
}

/// 依工具名稱建立工具實例
/// 需要 PgPool + bot_id 的工具（scratchpad）會使用傳入的參數
pub fn build_tool_by_name(
    name: &str,
    pg: &std::sync::Arc<sqlx::PgPool>,
    bot_id: &str,
) -> Option<Box<dyn Tool>> {
    match name {
        "save_memo" | "read_memo" => {
            // scratchpad::tools 回傳兩個，依名稱篩選
            scratchpad::tools(std::sync::Arc::clone(pg), bot_id.to_owned())
                .into_iter()
                .find(|t| t.definition().name == name)
        }
        "fetch" => Some(fetch::tool()),
        _ => None,
    }
}

/// 依工具名稱列表建立 ToolRegistry
pub fn build_registry(
    names: &[String],
    pg: &std::sync::Arc<sqlx::PgPool>,
    bot_id: &str,
) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for name in names {
        if let Some(tool) = build_tool_by_name(name, pg, bot_id) {
            registry.register(tool);
        }
    }
    registry
}

/// 工具註冊表，管理所有可用工具
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: vec![] }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn definitions(&self) -> Vec<ToolDef> {
        self.tools.iter().map(|t| t.definition()).collect()
    }

    pub async fn call(&self, name: &str, args: serde_json::Value) -> Result<String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.definition().name == name)
            .ok_or_else(|| anyhow::anyhow!("未知工具: {name}"))?;
        tool.call(args).await
    }
}
