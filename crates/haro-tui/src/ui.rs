use std::io;

use anyhow::Result;
use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::widgets::*;
use sqlx::PgPool;

use crate::db;

const PAGE_SIZE: i64 = 50;

/// 目前顯示的分頁
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Usage,
    Daily,
    Messages,
}

impl Tab {
    const ALL: [Tab; 3] = [Tab::Usage, Tab::Daily, Tab::Messages];

    fn title(self) -> &'static str {
        match self {
            Tab::Usage => "Token Usage",
            Tab::Daily => "Daily Summary",
            Tab::Messages => "Messages",
        }
    }

    fn index(self) -> usize {
        match self {
            Tab::Usage => 0,
            Tab::Daily => 1,
            Tab::Messages => 2,
        }
    }
}

struct App {
    pool: PgPool,
    tab: Tab,
    /// Agent 篩選（None = 全部）
    agents: Vec<String>,
    agent_idx: Option<usize>,

    // Usage tab
    usage_rows: Vec<db::TokenUsageRow>,
    usage_state: TableState,
    usage_offset: i64,

    // Daily tab
    daily_rows: Vec<db::DailySummary>,
    daily_state: TableState,
    daily_days: i32,

    // Messages tab
    msg_rows: Vec<db::MessageRow>,
    msg_state: TableState,
    msg_offset: i64,

    // Detail popup
    detail: Option<String>,

    should_quit: bool,
}

impl App {
    fn new(pool: PgPool) -> Self {
        Self {
            pool,
            tab: Tab::Usage,
            agents: Vec::new(),
            agent_idx: None,
            usage_rows: Vec::new(),
            usage_state: TableState::default(),
            usage_offset: 0,
            daily_rows: Vec::new(),
            daily_state: TableState::default(),
            daily_days: 30,
            msg_rows: Vec::new(),
            msg_state: TableState::default(),
            msg_offset: 0,
            detail: None,
            should_quit: false,
        }
    }

    fn agent_filter(&self) -> Option<&str> {
        self.agent_idx.map(|i| self.agents[i].as_str())
    }

    fn agent_label(&self) -> &str {
        self.agent_idx
            .map(|i| self.agents[i].as_str())
            .unwrap_or("all")
    }

    async fn load_agents(&mut self) -> Result<()> {
        self.agents = db::get_agents(&self.pool).await?;
        Ok(())
    }

    async fn load_current_tab(&mut self) -> Result<()> {
        match self.tab {
            Tab::Usage => self.load_usage().await,
            Tab::Daily => self.load_daily().await,
            Tab::Messages => self.load_messages().await,
        }
    }

    async fn load_usage(&mut self) -> Result<()> {
        self.usage_rows = db::get_token_usage(
            &self.pool,
            self.agent_filter(),
            PAGE_SIZE,
            self.usage_offset,
        )
        .await?;
        if self.usage_rows.is_empty() {
            self.usage_state.select(None);
        } else if self.usage_state.selected().is_none() {
            self.usage_state.select(Some(0));
        }
        Ok(())
    }

    async fn load_daily(&mut self) -> Result<()> {
        self.daily_rows =
            db::get_daily_summary(&self.pool, self.agent_filter(), self.daily_days).await?;
        if self.daily_rows.is_empty() {
            self.daily_state.select(None);
        } else if self.daily_state.selected().is_none() {
            self.daily_state.select(Some(0));
        }
        Ok(())
    }

    async fn load_messages(&mut self) -> Result<()> {
        self.msg_rows =
            db::get_messages(&self.pool, self.agent_filter(), PAGE_SIZE, self.msg_offset).await?;
        if self.msg_rows.is_empty() {
            self.msg_state.select(None);
        } else if self.msg_state.selected().is_none() {
            self.msg_state.select(Some(0));
        }
        Ok(())
    }

    fn move_selection(&mut self, delta: i32) {
        let (state, len) = match self.tab {
            Tab::Usage => (&mut self.usage_state, self.usage_rows.len()),
            Tab::Daily => (&mut self.daily_state, self.daily_rows.len()),
            Tab::Messages => (&mut self.msg_state, self.msg_rows.len()),
        };
        if len == 0 {
            return;
        }
        let cur = state.selected().unwrap_or(0) as i32;
        let next = (cur + delta).clamp(0, len as i32 - 1) as usize;
        state.select(Some(next));
    }

    async fn next_page(&mut self) -> Result<()> {
        match self.tab {
            Tab::Usage => {
                if self.usage_rows.len() as i64 == PAGE_SIZE {
                    self.usage_offset += PAGE_SIZE;
                    self.usage_state.select(Some(0));
                    self.load_usage().await?;
                }
            }
            Tab::Messages => {
                if self.msg_rows.len() as i64 == PAGE_SIZE {
                    self.msg_offset += PAGE_SIZE;
                    self.msg_state.select(Some(0));
                    self.load_messages().await?;
                }
            }
            Tab::Daily => {
                self.daily_days += 30;
                self.load_daily().await?;
            }
        }
        Ok(())
    }

    async fn prev_page(&mut self) -> Result<()> {
        match self.tab {
            Tab::Usage => {
                if self.usage_offset > 0 {
                    self.usage_offset = (self.usage_offset - PAGE_SIZE).max(0);
                    self.usage_state.select(Some(0));
                    self.load_usage().await?;
                }
            }
            Tab::Messages => {
                if self.msg_offset > 0 {
                    self.msg_offset = (self.msg_offset - PAGE_SIZE).max(0);
                    self.msg_state.select(Some(0));
                    self.load_messages().await?;
                }
            }
            Tab::Daily => {
                if self.daily_days > 30 {
                    self.daily_days -= 30;
                    self.load_daily().await?;
                }
            }
        }
        Ok(())
    }

    async fn cycle_agent(&mut self) -> Result<()> {
        if self.agents.is_empty() {
            return Ok(());
        }
        self.agent_idx = match self.agent_idx {
            None => Some(0),
            Some(i) if i + 1 >= self.agents.len() => None,
            Some(i) => Some(i + 1),
        };
        // Reset pagination
        self.usage_offset = 0;
        self.msg_offset = 0;
        self.usage_state.select(None);
        self.msg_state.select(None);
        self.daily_state.select(None);
        self.load_current_tab().await
    }

    fn show_detail(&mut self) {
        match self.tab {
            Tab::Usage => {
                if let Some(i) = self.usage_state.selected() {
                    if let Some(row) = self.usage_rows.get(i) {
                        self.detail = Some(format!(
                            "ID: {}\nAgent: {}\nModel: {}\nInput: {} tokens\nOutput: {} tokens\nTotal: {} tokens\nTime: {}",
                            row.id,
                            row.agent_id,
                            row.model,
                            row.input_tokens,
                            row.output_tokens,
                            row.input_tokens + row.output_tokens,
                            row.created_at
                                .with_timezone(&chrono::Local)
                                .format("%Y-%m-%d %H:%M:%S"),
                        ));
                    }
                }
            }
            Tab::Daily => {
                if let Some(i) = self.daily_state.selected() {
                    if let Some(row) = self.daily_rows.get(i) {
                        self.detail = Some(format!(
                            "Date: {}\nAgent: {}\nModel: {}\nCalls: {}\nInput: {} tokens\nOutput: {} tokens\nTotal: {} tokens\nAvg/call: {} tokens",
                            row.day.format("%Y-%m-%d"),
                            row.agent_id,
                            row.model,
                            row.call_count,
                            row.total_input,
                            row.total_output,
                            row.total_input + row.total_output,
                            if row.call_count > 0 {
                                (row.total_input + row.total_output) / row.call_count
                            } else {
                                0
                            },
                        ));
                    }
                }
            }
            Tab::Messages => {
                if let Some(i) = self.msg_state.selected() {
                    if let Some(row) = self.msg_rows.get(i) {
                        self.detail = Some(format!(
                            "ID: {}\nAgent: {}\nTokens: {}\nTime: {}\n\n--- Summary ---\n{}\n\n--- Original ---\n{}",
                            row.id,
                            row.bot_id,
                            row.token_count,
                            row.created_at
                                .with_timezone(&chrono::Local)
                                .format("%Y-%m-%d %H:%M:%S"),
                            row.dense_summary,
                            row.original_text,
                        ));
                    }
                }
            }
        }
    }
}

fn draw(frame: &mut Frame, app: &mut App) {
    let chunks = Layout::vertical([
        Constraint::Length(3), // tabs + agent filter
        Constraint::Min(0),    // main content
        Constraint::Length(1), // status bar
    ])
    .split(frame.area());

    // ── Tab bar ──
    let titles: Vec<Line> = Tab::ALL
        .iter()
        .map(|t| {
            let style = if *t == app.tab {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            Line::from(t.title()).style(style)
        })
        .collect();

    let tabs = Tabs::new(titles)
        .select(app.tab.index())
        .divider(" │ ")
        .block(
            Block::bordered()
                .title(format!(" haro-tui  agent: {} ", app.agent_label()))
                .title_alignment(Alignment::Left),
        );
    frame.render_widget(tabs, chunks[0]);

    // ── Main content ──
    match app.tab {
        Tab::Usage => draw_usage(frame, app, chunks[1]),
        Tab::Daily => draw_daily(frame, app, chunks[1]),
        Tab::Messages => draw_messages(frame, app, chunks[1]),
    }

    // ── Status bar ──
    let page_info = match app.tab {
        Tab::Usage => format!(
            "rows: {} | offset: {}",
            app.usage_rows.len(),
            app.usage_offset
        ),
        Tab::Daily => format!("rows: {} | days: {}", app.daily_rows.len(), app.daily_days),
        Tab::Messages => format!("rows: {} | offset: {}", app.msg_rows.len(), app.msg_offset),
    };
    let status = Line::from(vec![
        " q".fg(Color::Yellow),
        " quit  ".into(),
        "Tab".fg(Color::Yellow),
        " switch  ".into(),
        "a".fg(Color::Yellow),
        " agent  ".into(),
        "↑↓".fg(Color::Yellow),
        " navigate  ".into(),
        "←→".fg(Color::Yellow),
        " page  ".into(),
        "Enter".fg(Color::Yellow),
        " detail  ".into(),
        "r".fg(Color::Yellow),
        " refresh  ".into(),
        format!("│ {page_info}").fg(Color::DarkGray),
    ]);
    frame.render_widget(status, chunks[2]);

    // ── Detail popup ──
    if let Some(text) = &app.detail {
        let area = centered_rect(70, 70, frame.area());
        frame.render_widget(Clear, area);
        let detail = Paragraph::new(text.as_str())
            .wrap(Wrap { trim: false })
            .block(
                Block::bordered()
                    .title(" Detail (Esc to close) ")
                    .border_style(Style::default().fg(Color::Cyan)),
            );
        frame.render_widget(detail, area);
    }
}

fn draw_usage(frame: &mut Frame, app: &mut App, area: Rect) {
    let header = Row::new(["Time", "Agent", "Model", "Input", "Output", "Total"])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(0);

    let rows: Vec<Row> = app
        .usage_rows
        .iter()
        .map(|r| {
            let total = r.input_tokens + r.output_tokens;
            Row::new([
                r.created_at
                    .with_timezone(&chrono::Local)
                    .format("%m-%d %H:%M")
                    .to_string(),
                r.agent_id.clone(),
                r.model.clone(),
                format_tokens(r.input_tokens),
                format_tokens(r.output_tokens),
                format_tokens(total),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(12),
        Constraint::Length(16),
        Constraint::Min(20),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::bordered().title(" Token Usage (per call) "))
        .row_highlight_style(Style::default().bg(Color::DarkGray));

    frame.render_stateful_widget(table, area, &mut app.usage_state);
}

fn draw_daily(frame: &mut Frame, app: &mut App, area: Rect) {
    let header = Row::new([
        "Date", "Agent", "Model", "Calls", "Input", "Output", "Total",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );

    let rows: Vec<Row> = app
        .daily_rows
        .iter()
        .map(|r| {
            let total = r.total_input + r.total_output;
            Row::new([
                r.day.format("%Y-%m-%d").to_string(),
                r.agent_id.clone(),
                r.model.clone(),
                r.call_count.to_string(),
                format_tokens_i64(r.total_input),
                format_tokens_i64(r.total_output),
                format_tokens_i64(total),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(11),
        Constraint::Length(16),
        Constraint::Min(20),
        Constraint::Length(7),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::bordered().title(format!(" Daily Summary (last {} days) ", app.daily_days)))
        .row_highlight_style(Style::default().bg(Color::DarkGray));

    frame.render_stateful_widget(table, area, &mut app.daily_state);
}

fn draw_messages(frame: &mut Frame, app: &mut App, area: Rect) {
    let header = Row::new(["Time", "Agent", "Tokens", "Summary"]).style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );

    let rows: Vec<Row> = app
        .msg_rows
        .iter()
        .map(|r| {
            let summary = r.dense_summary.lines().next().unwrap_or("").to_owned();
            let summary = if summary.len() > 80 {
                format!("{}…", &summary[..79])
            } else {
                summary
            };
            Row::new([
                r.created_at
                    .with_timezone(&chrono::Local)
                    .format("%m-%d %H:%M")
                    .to_string(),
                r.bot_id.clone(),
                format_tokens(r.token_count),
                summary,
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(12),
        Constraint::Length(16),
        Constraint::Length(8),
        Constraint::Min(30),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::bordered().title(" Messages "))
        .row_highlight_style(Style::default().bg(Color::DarkGray));

    frame.render_stateful_widget(table, area, &mut app.msg_state);
}

fn format_tokens(n: i32) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_tokens_i64(n: i64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(area);

    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(popup_layout[1])[1]
}

pub async fn run(pool: PgPool) -> Result<()> {
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(pool);
    app.load_agents().await?;
    app.load_current_tab().await?;

    loop {
        terminal.draw(|f| draw(f, &mut app))?;

        if event::poll(std::time::Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                // Detail popup open — only Esc closes it
                if app.detail.is_some() {
                    if key.code == KeyCode::Esc {
                        app.detail = None;
                    }
                    continue;
                }

                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('c')
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            || key.code == KeyCode::Char('q') =>
                    {
                        app.should_quit = true;
                    }
                    KeyCode::Tab => {
                        app.tab = match app.tab {
                            Tab::Usage => Tab::Daily,
                            Tab::Daily => Tab::Messages,
                            Tab::Messages => Tab::Usage,
                        };
                        app.load_current_tab().await?;
                    }
                    KeyCode::BackTab => {
                        app.tab = match app.tab {
                            Tab::Usage => Tab::Messages,
                            Tab::Daily => Tab::Usage,
                            Tab::Messages => Tab::Daily,
                        };
                        app.load_current_tab().await?;
                    }
                    KeyCode::Char('a') => {
                        app.cycle_agent().await?;
                    }
                    KeyCode::Up | KeyCode::Char('k') => app.move_selection(-1),
                    KeyCode::Down | KeyCode::Char('j') => app.move_selection(1),
                    KeyCode::Right | KeyCode::Char('l') => app.next_page().await?,
                    KeyCode::Left | KeyCode::Char('h') => app.prev_page().await?,
                    KeyCode::Enter => app.show_detail(),
                    KeyCode::Char('r') => app.load_current_tab().await?,
                    KeyCode::Home | KeyCode::Char('g') => app.move_selection(i32::MIN / 2),
                    KeyCode::End | KeyCode::Char('G') => app.move_selection(i32::MAX / 2),
                    _ => {}
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}
