use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;

use crate::config;

#[derive(Args)]
pub struct InitAgentArgs {
    /// Agent ID（即目錄名稱），未提供則互動詢問
    pub name: Option<String>,

    /// agents 根目錄（預設 ~/.local/share/haro/agents）
    #[arg(long)]
    pub base: Option<PathBuf>,
}

#[derive(Args)]
pub struct AddConfigArgs {
    /// Agent ID（即目錄名稱），未提供則互動詢問
    pub name: Option<String>,

    /// agents 根目錄（預設 ~/.local/share/haro/agents）
    #[arg(long)]
    pub base: Option<PathBuf>,
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

    // 解析 agents 根目錄：--base > config agents_path > default
    let base = args.base.unwrap_or_else(|| {
        config::AppConfig::load()
            .ok()
            .map(|c| c.agents_dir())
            .unwrap_or_else(config::default_agents_dir)
    });

    let agent_dir = base.join(&name);
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

    // 4. Cron 排程
    eprintln!();
    let create_cron = ask_default("是否建立範例 cron.toml？(y/n)", "n")?;
    let cron_content = if create_cron.starts_with('y') || create_cron.starts_with('Y') {
        include_str!("../templates/cron.toml").to_owned()
    } else {
        String::new()
    };

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
    let config_content =
        format!("[[channels]]\nchannel_type = \"{channel_type}\"\n{token_line}\n{users_line}\n");
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
    if !cron_content.is_empty() {
        write_file(&agent_dir, "cron.toml", &cron_content)?;
        files.push("cron.toml");
    }

    eprintln!("\n✅ Agent「{name}」已建立: {}", agent_dir.display());
    for f in &files {
        eprintln!("   {f}");
    }

    Ok(())
}

pub fn add_config(args: AddConfigArgs) -> Result<()> {
    eprintln!("=== 為現有 Agent 新增設定檔 ===\n");

    let base = args.base.unwrap_or_else(|| {
        config::AppConfig::load()
            .ok()
            .map(|c| c.agents_dir())
            .unwrap_or_else(config::default_agents_dir)
    });

    let name = match args.name {
        Some(n) => n,
        None => {
            // 列出現有 agents
            let mut agents: Vec<String> = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&base) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        if let Some(name) = entry.file_name().to_str() {
                            agents.push(name.to_owned());
                        }
                    }
                }
            }
            agents.sort();
            if agents.is_empty() {
                anyhow::bail!("在 {} 下找不到任何 agent", base.display());
            }
            eprintln!("可用的 Agents:");
            for (i, a) in agents.iter().enumerate() {
                eprintln!("  {}) {a}", i + 1);
            }
            let choice = ask("選擇 Agent（輸入編號或名稱）")?;
            if let Ok(idx) = choice.parse::<usize>() {
                if idx >= 1 && idx <= agents.len() {
                    agents[idx - 1].clone()
                } else {
                    anyhow::bail!("無效的編號: {idx}");
                }
            } else {
                choice
            }
        }
    };

    let agent_dir = base.join(&name);
    if !agent_dir.exists() {
        anyhow::bail!("Agent 目錄不存在: {}", agent_dir.display());
    }

    let mut created = Vec::new();

    // HEARTBEAT.md
    let hb_path = agent_dir.join("HEARTBEAT.md");
    if hb_path.exists() {
        eprintln!("HEARTBEAT.md 已存在，跳過");
    } else {
        let content = ask_md("HEARTBEAT.md（心跳提示詞）")?;
        if !content.is_empty() {
            write_file(&agent_dir, "HEARTBEAT.md", &content)?;
            created.push("HEARTBEAT.md");
        }
    }

    // BRAIN_HEARTBEAT.md
    let bh_path = agent_dir.join("BRAIN_HEARTBEAT.md");
    if bh_path.exists() {
        eprintln!("BRAIN_HEARTBEAT.md 已存在，跳過");
    } else {
        let content = ask_md("BRAIN_HEARTBEAT.md（大腦心跳提示詞）")?;
        if !content.is_empty() {
            write_file(&agent_dir, "BRAIN_HEARTBEAT.md", &content)?;
            created.push("BRAIN_HEARTBEAT.md");
        }
    }

    // cron.toml
    let cron_path = agent_dir.join("cron.toml");
    if cron_path.exists() {
        eprintln!("cron.toml 已存在，跳過");
    } else {
        let create_cron = ask_default("是否建立範例 cron.toml？(y/n)", "n")?;
        if create_cron.starts_with('y') || create_cron.starts_with('Y') {
            let content = include_str!("../templates/cron.toml");
            write_file(&agent_dir, "cron.toml", content)?;
            created.push("cron.toml");
        }
    }

    if created.is_empty() {
        eprintln!("\n沒有建立任何新檔案。");
    } else {
        eprintln!("\n✅ 已為 Agent「{name}」建立:");
        for f in &created {
            eprintln!("   {f}");
        }
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
    Ok(if v.is_empty() {
        None
    } else {
        Some(v.to_owned())
    })
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
