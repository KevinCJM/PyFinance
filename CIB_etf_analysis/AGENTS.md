# Repository Guidelines

## Project Structure & Module Organization
- Root-level scripts drive the workflow (e.g., `A01_get_data.py`).
- Data artifacts are currently written to the repo root (e.g., `etf_info_df.xlsx`, `etf_daily_df.parquet`). Prefer writing new outputs to `data/` (e.g., `data/raw/`, `data/processed/`).
- If adding reusable code, place it in `src/` with package-style modules; keep notebooks (if any) under `notebooks/`.

## Build, Test, and Development Commands
- Python setup
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install pandas tushare pyarrow pytest black ruff`
- Run locally
  - `export TUSHARE_TOKEN=...` (avoid hardcoding tokens)
  - `python A01_get_data.py` (fetches ETF info and writes outputs)
- Format & lint
  - `black .` (format) • `ruff .` (lint)
- Tests
  - `pytest -q`

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; 88-char line length.
- Use `black` for formatting and `ruff` for linting; fix warnings before opening a PR.
- Prefer type hints and docstrings for public functions.
- File naming: follow existing numeric prefix pattern for scripts (e.g., `A02_<verb>_<topic>.py`). Modules in `src/<package>/...` use snake_case files and PascalCase classes.

## Testing Guidelines
- Framework: `pytest`.
- Location: `tests/` mirroring module paths; name files `test_<module>.py`.
- Aim for meaningful unit tests around data transforms; add lightweight integration tests for API calls using fixtures/mocks.
- Run `pytest -q`; target basic coverage for new code and avoid network-dependent tests in CI.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood (e.g., `add`, `fix`, `refactor`). Example: `feat: add ETF daily fetch with parquet output`.
- PRs: include purpose, summary of changes, how to run/verify, and any follow-ups. Link issues and include sample output or screenshots when relevant.
- Ensure no secrets (API tokens) or large generated files are committed. Do not commit Excel/Parquet outputs (`*.xlsx`, `*.xls`, `*.parquet`). Prefer writing outputs under `data/` and keep them ignored via `.gitignore`.

## Security & Configuration Tips
- Do not hardcode tokens. Use environment variables (e.g., `TUSHARE_TOKEN`) or a local `.env` not committed to git.
- Be mindful of data size; avoid committing large exports (CSV/XLSX/Parquet). Consider writing to `data/processed/` and documenting paths.

## Agent-Specific Instructions
- 交流语言：默认使用中文进行沟通与说明。
- 自测与提交：提交代码前先本地自测（`pytest -q`、`ruff .`、`black --check .`）。若自测通过，自动执行 `git add -A && git commit -m "chore: update after self-test"` 并按需推送。
- 产物管理：Excel/Parquet 等数据产物不纳入版本控制（通过 `.gitignore` 忽略）。
