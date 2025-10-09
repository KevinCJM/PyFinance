# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI services in `app.py`, modeling utilities (`fit.py`, `optimizer.py`), and CLI entrypoints in `run.py`. Serves built frontend assets from `frontend/dist`.
- `frontend/`: Vite + React + Tailwind source under `src/`; compiled output in `dist/`.
- `data/`: Local Parquet/CSV inputs and fixtures. Keep only small, deterministic fixtures under version control.
- Root: shared utilities (e.g., `T01_get_data.py`), configuration (`config.py`), and documentation such as this guide.

## Build, Test, and Development Commands
- Backend setup: `cd backend && pip install -r requirements.txt`.
- Backend dev server: `uvicorn app:app --reload --host 0.0.0.0 --port 8000` for live API reloads.
- Unified demo: `python backend/run.py` after building the frontend.
- Frontend setup: `cd frontend && npm install`.
- Frontend dev server: `npm run dev` (proxies `/api` to the backend).
- Frontend build: `npm run build` to populate `frontend/dist`.

## Coding Style & Naming Conventions
- Python: follow PEP 8, 4-space indents, `snake_case` functions/variables, `PascalCase` Pydantic models. Keep HTTP route definitions in `backend/app.py`; place pure logic in dedicated modules. Use `DATA_DIR` helpers over hard-coded paths.
- TypeScript/React: functional components, PascalCase filenames (e.g., `ManualConstruction.tsx`), colocate styles and tests near components. Tailwind utility classes go directly in JSX.

## Testing Guidelines
- Backend: `pytest` under `backend/tests/`, files named `test_*.py`. Start with route 200/JSON shape checks and fixtures in `data/fixtures/`.
- Frontend: Vitest with React Testing Library (where present); colocate `*.test.tsx` beside the component.
- Aim for deterministic tests—no network calls. Prefer fixtures or mocks for market data.

## Commit & Pull Request Guidelines
- Use clear, scoped commit messages. Conventional prefixes (`feat:`, `fix:`, `docs:`, `refactor:`) keep history readable.
- PRs should explain purpose, highlight major changes, list run commands, and include UI screenshots when applicable. Note any data or token prerequisites.

## Security & Configuration Tips
- Never commit secrets. `config.py` must read tokens from environment variables or `.env` (kept out of git).
- CORS is permissive for development only; tighten origins before production deployment.
- Prefer running the backend with the virtualenv at `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`.

## Agent-Specific Instructions
- Communicate in Chinese with collaborators by default; keep code identifiers and comments in each language's standard style.
