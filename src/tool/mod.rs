pub mod scratchpad;

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
