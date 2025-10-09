# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI service entrypoints. `app.py` hosts routes, `run.py` powers autoreload dev server. Shared logic resides under `backend/services/` (e.g., `data_service.py`, `analysis_service.py`, `tushare_get_data.py`).
- `backend/data/`: Parquet cache and analysis outputs; `.meta/meta_index.json` tracks incremental fetches. Keep large artifacts out of version control.
- `frontend/`: React + TypeScript app scaffolded with Vite. Source lives in `frontend/src/` (`components/` for reusable UI, `pages/` for routed views). Build output is `frontend/dist/`.
- Tests: place backend tests in `backend/tests/test_*.py`; frontend tests under `frontend/src/__tests__/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt`: create/activate virtualenv and install backend deps.
- `python backend/run.py`: launch FastAPI dev server at `http://127.0.0.1:8000` with auto-reload.
- `curl :8000/api/health`: quick backend health check; refresh data via `curl -X POST :8000/api/stocks_basic/refresh`.
- `cd frontend && npm install && npm run dev`: install frontend packages and start Vite dev server.
- `npm run build`: produce production bundle in `frontend/dist/`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent, `snake_case` functions/modules, `PascalCase` classes. Add type hints where practical and prefer logging over prints.
- TypeScript/React: ESLint + Prettier via `frontend/eslint.config.js`; `PascalCase` components, `camelCase` hooks/vars. Favor Tailwind utility classes for styling.
- Centralize reusable backend logic in `backend/services/` and keep API payloads concise.

## Testing Guidelines
- Backend: run `pytest` from repo root; stub network interactions where needed for determinism.
- Frontend: use `vitest` + React Testing Library; co-locate specs in `frontend/src/__tests__/`.
- Ensure new features include regression coverage and mimic naming conventions (`test_<feature>.py`).

## Commit & Pull Request Guidelines
- Commit messages follow `<scope>: <summary>` (e.g., `frontend: unify ABC panels`). Keep scopes narrow and imperative.
- PRs should describe changes, link issues, note API/UI impacts, and attach screenshots for UI updates. Highlight data migrations or configuration steps.

## Security & Configuration Tips
- Set `TUSHARE_TOKEN` via `config.py` or environment; never commit secrets.
- Development CORS is open; tighten settings before production releases.
- Exclude heavy artifacts such as `backend/data/`, `frontend/dist/`, and `node_modules/` from commits.

## Agent-Specific Instructions
- Adhere to the established module layout when adding endpoints or shared UI assets.
- Update this guide if you introduce significant architectural or workflow changes so future agents inherit the latest process.
