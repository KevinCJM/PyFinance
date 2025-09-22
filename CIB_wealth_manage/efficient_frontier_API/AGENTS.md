# Repository Guidelines

This repository implements efficient-frontier and portfolio utilities for CIB use cases. Modules are organized as runnable Python scripts with shared tooling.

## Project Structure & Module Organization
- Root scripts
  - `A01_main_api.py` — end-to-end efficient frontier and plotting.
  - `A02_risk_boundaries_api.py` — risk boundary search (min/max risk).
  - `A03_ideal_portfolio_api.py` — ideal/target portfolio utilities.
  - `A04_construct_category_yield.py` — build category returns from index inputs.
  - `B01_cal_all_weights.py` — batch weight generation/calculation.
  - `T01_*`, `T02_*`, `T03_*`, `T04_*`, `T05_*` — shared tools (random weights, IO/logging, limits, plotting, DB utils).
- Data & samples
  - `sample_*.json` — runnable input examples.
  - Excel sources: `历史净值数据.xlsx`, `万得指数数据.xlsx`.
  - Output example: `efficient_frontier.html`.
- Dependencies: `requirements.txt`

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run main flow: `python A01_main_api.py` (uses `sample_A01_input_market.json` by default).
- Risk boundaries: `python A02_risk_boundaries_api.py`
- Construct category returns: `python A04_construct_category_yield.py`
Notes
- Excel engines: install `openpyxl` (xlsx) and `xlrd<2.0` (xls) per `T02_other_tools.read_excel_compat`.
- Numba is optional; code falls back if unavailable.
 - Docker runtime: image `python36:3.6.9` (Python 3.6.9). Example: `docker run -it --rm -v "$PWD:/app" -w /app python36:3.6.9 bash`.

## Coding Style & Naming Conventions
- Python 3.6.9; 遵循 PEP 8，4 空格缩进。
- Filenames use prefixes: `A..` (APIs), `B..` (batch), `T..` (tools).
- Functions/variables: `snake_case`; constants: `UPPER_SNAKE`; classes: `PascalCase`.
- Prefer type hints (`typing`) and numpy dtypes (`float32` for arrays where possible).

## Testing Guidelines
- No formal test suite yet. Add tests under `tests/` as `test_*.py` using `pytest`.
- Smoke tests: run scripts with provided `sample_*.json` and verify JSON outputs and generated plots.
- Aim for coverage of constraint handling, VaR/vol modes, and Excel/JSON data paths.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject; Chinese or English allowed. Example: `feat(A01): add VaR paramization` or `修复: Excel 引擎回退逻辑`.
- PRs must include:
  - What/why summary and linked issue (if any).
  - Sample input JSON and steps to reproduce locally.
  - Before/after screenshots or the generated `efficient_frontier.html` when plotting changes.
  - Performance notes for changes affecting sampling/numba paths.

## Security & Configuration Tips
- Do not commit real customer data. Use `sample_*.json` and sanitized Excel files.
- Large computations: prefer deterministic seeds (`RANDOM_SEED`) and document parameter changes (`ROUNDS_CONFIG`, `VAR_PARAMS`).
