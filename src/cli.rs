use std::io::{self, Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use clap::Args;

#[derive(Args)]
pub struct InitAgentArgs {
    /// Agent ID（即目錄名稱），未提供則互動詢問
    pub name: Option<String>,

    /// agents 根目錄（預設 "agents"）
    #[arg(long, default_value = "agents")]
    pub base: String,
}

pub fn init_agent(args: InitAgentArgs) -> Result<()> {
    eprintln!("=== Haro Agent 建立精靈 ===\n");

    // 1. Agent ID
    let name = match args.name {
        Some(n) => n,
        None => ask("Agent ID（目錄名稱）")?,
    };
    if name.is_empty() {
        anyhow::bail!("Agent ID 不可為空");
    }

    let agent_dir = Path::new(&args.base).join(&name);
    if agent_dir.exists() {
        anyhow::bail!("Agent 目錄已存在: {}", agent_dir.display());
    }

    // 2. Channel
    let channel_type = ask_default("Channel 類型", "telegram")?;

    let bot_token_env = ask_optional("Bot token 環境變數名稱（留空跳過）")?;

    let allowed_users = ask_optional("允許的使用者（逗號分隔 user ID 或 @username，留空不限制）")?;

    // 3. Markdown 檔案
    eprintln!();
    let soul = ask_md("SOUL.md（性格設定）")?;
    let prompt = ask_md("PROMPT.md（自訂提示詞）")?;
    let heartbeat = ask_md("HEARTBEAT.md（心跳提示詞）")?;
    let brain_heartbeat = ask_md("BRAIN_HEARTBEAT.md（大腦心跳提示詞）")?;

    // 建立目錄和檔案
    std::fs::create_dir_all(&agent_dir)
        .with_context(|| format!("無法建立目錄: {}", agent_dir.display()))?;

    // config.toml
    let token_line = match &bot_token_env {
        Some(env) => format!("bot_token_env = \"{env}\""),
        None => "# bot_token_env = \"BOT_TOKEN\"".into(),
    };
    let users_line = match &allowed_users {
        Some(users) => {
            let items: Vec<String> = users
                .split(',')
                .map(|s| format!("\"{}\"", s.trim()))
                .collect();
            format!("allowed_users = [{}]", items.join(", "))
        }
        None => "allowed_users = []".into(),
    };
    let config_content = format!(
        "[[channels]]\nchannel_type = \"{channel_type}\"\n{token_line}\n{users_line}\n"
    );
    write_file(&agent_dir, "config.toml", &config_content)?;

    write_file(&agent_dir, "SOUL.md", &soul)?;
    write_file(&agent_dir, "PROMPT.md", &prompt)?;

    let mut files = vec!["config.toml", "SOUL.md", "PROMPT.md"];
    if !heartbeat.is_empty() {
        write_file(&agent_dir, "HEARTBEAT.md", &heartbeat)?;
        files.push("HEARTBEAT.md");
    }
    if !brain_heartbeat.is_empty() {
        write_file(&agent_dir, "BRAIN_HEARTBEAT.md", &brain_heartbeat)?;
        files.push("BRAIN_HEARTBEAT.md");
    }

    eprintln!("\n✅ Agent「{name}」已建立: {}", agent_dir.display());
    for f in &files {
        eprintln!("   {f}");
    }

    Ok(())
}

/// 詢問必填欄位
fn ask(label: &str) -> Result<String> {
    eprint!("{label}: ");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(line.trim().to_owned())
}

/// 詢問欄位（有預設值）
fn ask_default(label: &str, default: &str) -> Result<String> {
    eprint!("{label} [{default}]: ");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let v = line.trim();
    Ok(if v.is_empty() {
        default.to_owned()
    } else {
        v.to_owned()
    })
}

/// 詢問可選欄位（留空回傳 None）
fn ask_optional(label: &str) -> Result<Option<String>> {
    eprint!("{label}: ");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let v = line.trim();
    Ok(if v.is_empty() { None } else { Some(v.to_owned()) })
}

/// 詢問 Markdown 檔案內容，提供三種選項
fn ask_md(label: &str) -> Result<String> {
    eprint!("{label}\n  1) 跳過（留空）\n  2) 直接輸入\n  3) 開啟編輯器\n  選擇 [1]: ");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;

    match line.trim() {
        "" | "1" => Ok(String::new()),
        "2" => read_inline(label),
        "3" => open_editor(label),
        other => {
            eprintln!("無效的選項「{other}」，跳過");
            Ok(String::new())
        }
    }
}

/// 從 stdin 讀取多行內容
fn read_inline(label: &str) -> Result<String> {
    eprintln!("請輸入 {label}（輸入空行結束）:");
    let mut lines = Vec::new();
    let stdin = io::stdin();
    loop {
        let mut line = String::new();
        stdin.read_line(&mut line)?;
        if line.trim().is_empty() {
            break;
        }
        lines.push(line);
    }
    Ok(lines.join("").trim_end().to_owned())
}

/// 開啟 $EDITOR 讓使用者編輯暫存檔
fn open_editor(label: &str) -> Result<String> {
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".into());

    let tmp_path = std::env::temp_dir().join(format!("haro_{}.md", label.replace('.', "_")));
    std::fs::write(&tmp_path, "")?;

    let status = std::process::Command::new(&editor)
        .arg(&tmp_path)
        .status()
        .with_context(|| format!("無法啟動編輯器: {editor}"))?;

    if !status.success() {
        anyhow::bail!("編輯器非正常結束");
    }

    let mut content = String::new();
    std::fs::File::open(&tmp_path)?.read_to_string(&mut content)?;
    let _ = std::fs::remove_file(&tmp_path);

    Ok(content.trim().to_owned())
}

fn write_file(dir: &Path, name: &str, content: &str) -> Result<()> {
    let path = dir.join(name);
    let mut f =
        std::fs::File::create(&path).with_context(|| format!("無法建立 {}", path.display()))?;
    f.write_all(content.as_bytes())?;
    if !content.is_empty() && !content.ends_with('\n') {
        f.write_all(b"\n")?;
    }
    Ok(())
}
