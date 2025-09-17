# Repository Guidelines

This repository implements efficient-frontier APIs and utilities in Python. Use the scripts under the root to compute random-walk portfolios, risk boundaries, and plotting outputs backed by Excel time series.

## Project Structure & Module Organization
- `A01_main_api.py` — main workflow: parse JSON, load Excel, compute frontiers, optional plotting.
- `A02_risk_boundaries_api.py` — find min/max risk portfolios under constraints.
- `A03_ideal_portfolio_api.py` — scaffolding for ideal portfolio suggestions (WIP) using existing EF data.
- `T01_generate_random_weights.py` — core algorithms: random walk, constraints, metrics, optional CVX/SLSQP refine.
- `T02_other_tools.py` — utilities: logging and Excel loaders.
- `T03_weight_limit_cal.py` — weight limit derivation for standard levels and holdings.
- `T04_show_plt.py` — Plotly-based visualization helpers.
- Data: `历史净值数据_万得指数.xlsx` (sheet `历史净值数据`) and variants; docs: `Z01_API接口说明文档.md`.

## Build, Test, and Development Commands
- Setup (Python 3.10+ recommended):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install numpy pandas plotly scipy numba cvxpy openpyxl`
- Run main API locally: `python A01_main_api.py` (uses in-file demo JSON). Set `draw_plt_filename` to save HTML instead of opening a window.
- Risk boundaries demo: `python A02_risk_boundaries_api.py`.
- Lint/format (optional): `pip install black ruff && black . && ruff check .`

## Coding Style & Naming Conventions
- Python style: 4-space indent, type hints, snake_case for vars/functions, UPPER_CASE for constants.
- Keep module prefixes: `A0x_` for API entrypoints, `T0x_` for tools, `Z0x_` for docs.
- Imports are intra-repo (same folder). Do not rearrange file locations without updating imports.
- Prefer `T02_other_tools.log` for simple timestamped logs.

## Testing Guidelines
- No formal test suite yet. Validate changes via the `__main__` demos in `A01_`/`A02_`.
- If adding tests, use `pytest` with names `test_*.py` at repo root; run `pytest -q`.
- Provide small fixtures or sample JSON payloads; avoid committing large proprietary data.

## Commit & Pull Request Guidelines
- Commits: concise, present tense. Prefer Conventional Commits (e.g., `feat: add SLSQP refine`, `fix: VaR negative clipping`).
- PRs: include purpose, sample JSON input, expected key outputs (e.g., weights/risk ranges), performance notes, and a screenshot or saved HTML of plots when applicable. Link related issues/tasks.

## Security & Configuration Tips
- Data files may be proprietary. Do not commit sensitive spreadsheets; use local paths and document sheet names.
- Some features are optional and heavy (`cvxpy`, `numba`, `scipy`); code degrades gracefully if missing, but install them for full functionality and speed.

## Agent-Specific Instructions
- Default language: Chinese. The agent should respond in Chinese for this repository unless asked otherwise.
- Long-term preferences file: `.agent_memory.json` at repo root. Store lightweight, non-sensitive settings (e.g., `preferred_language`, small notes). Do not store secrets or large data.
- If the file is missing, create it on first write; keep JSON flat and human-editable. Example:
  - `{ "preferred_language": "zh-CN", "style": "concise" }`
