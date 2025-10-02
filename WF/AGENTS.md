# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI service. Key files: `app.py`, `run.py`, `requirements.txt`.
- `backend/services/`: data and analysis utilities (`data_service.py`, `analysis_service.py`).
- `backend/data/`: cached parquet files (created at runtime). Do not commit large data.
- `frontend/`: React + TypeScript + Vite app. Build output in `frontend/dist`.

## Build, Test, and Development Commands
- Backend setup: `python -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt`.
- Run backend (auto-reload + opens browser): `python backend/run.py` (serves API on `http://127.0.0.1:8000`).
- Frontend dev: `cd frontend && npm install && npm run dev`.
- Frontend build: `cd frontend && npm run build` (backend serves `frontend/dist` at `/`).
- Quick API checks: `curl http://127.0.0.1:8000/api/health`, `curl 'http://127.0.0.1:8000/api/stocks?page=1&page_size=5'`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent, type hints where practical. `snake_case` for files/functions, `PascalCase` for classes. Keep services in `backend/services/`.
- TypeScript/React: follow ESLint config in `frontend/eslint.config.js`. `PascalCase` components, `camelCase` vars/functions. One component per file under `src/`.
- Logs over prints; avoid committing notebooks or data dumps.

## Testing Guidelines
- No test framework is configured yet. Preferred: `pytest` for backend (`backend/tests/test_*.py`).
- Frontend: add `vitest` + React Testing Library; place tests under `frontend/src/__tests__/`.
- Keep tests fast and deterministic; aim for critical-path coverage (services, API handlers, UI state).

## Commit & Pull Request Guidelines
- Commit format: `<scope>: <summary>` (e.g., `backend: add A-point API`). Chinese or English is fine; be specific and imperative.
- Include rationale and user impact in bodies when non-trivial.
- PRs: clear description, linked issues, repro steps, screenshots for UI, and notes on data/migration. Keep diffs focused.

## Security & Configuration Tips
- CORS is `*` for dev; restrict origins for production.
- `akshare` triggers network calls; prefer cached parquet in `backend/data/` for local dev.
- Do not commit credentials or large `.parquet` files.

## Agent-Specific Instructions
- Scope: this file applies to the whole repo. Prefer minimal, targeted changes; avoid broad refactors. Mirror existing patterns, update docs when endpoints or commands change.
