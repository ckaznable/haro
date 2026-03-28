# Repository Guidelines

## Project Structure & Module Organization
This repository is a Rust workspace with two crates:
- `crates/haro`: main runtime (agent orchestration, channels, tools, DB, search).
- `crates/haro-tui`: terminal UI for inspecting PostgreSQL data.

Key directories:
- `crates/haro/src/{agent,api,channel,db,runner,tool}` for feature modules.
- `crates/haro/migrations` for SQL migrations.
- `crates/haro/templates` for starter `config.toml`, `cron.toml`, and `cmd.toml`.
- `deploy/` for local container orchestration (`docker-compose.yml`, pod manifests).
- `scripts/build-release.sh` for release binary/image builds.

## Build, Test, and Development Commands
- `cargo build --workspace`: build all crates in dev mode.
- `cargo run -p haro -- init`: generate a starter `config.toml`.
- `cargo run -p haro -- init-agent <name>`: scaffold an agent directory.
- `cargo run -p haro -- cli`: run LLM CLI mode without starting channels.
- `cargo run -p haro-tui -- --database-url <postgres-url>`: launch DB TUI.
- `cargo test --workspace`: run unit/async tests across the workspace.
- `cargo fmt --all` and `cargo clippy --workspace --all-targets -D warnings`: format and lint before PR.
- `./scripts/build-release.sh --image`: build release binaries and container image.

## Coding Style & Naming Conventions
Use standard Rust style (`rustfmt`): 4-space indentation, snake_case for modules/functions, PascalCase for types, SCREAMING_SNAKE_CASE for constants. Keep modules focused by domain (for example, new tools belong under `src/tool/`). Prefer `anyhow::Result` for app-level flows and `thiserror` for typed errors where needed.

## Testing Guidelines
Tests are colocated with code using `#[cfg(test)] mod tests` and async tests use `#[tokio::test]`. Follow existing naming such as `test_<behavior>` for clarity. Add or update tests for behavior changes in parsing, tool dispatch, channel routing, and search logic. There is no enforced coverage threshold; treat regression prevention as the baseline.

## Commit & Pull Request Guidelines
Follow the repository’s Conventional Commit pattern seen in history: `feat:`, `fix:`, `refactor:`, `chore:`. Keep commits scoped and atomic.

PRs should include:
- A concise problem/solution summary.
- Linked issue or context.
- Commands run locally (build, fmt, clippy, tests).
- Screenshots or terminal snippets for user-facing behavior changes (especially TUI/CLI output).

## Security & Configuration Tips
Do not commit secrets. Use `api_key_env` and environment variables (`GEMINI_API_KEY`, `DATABASE_URL`, `QDRANT_URL`) instead of hardcoding credentials. Prefer `HARO_CONFIG` for explicit config selection in local/dev environments.
