# Repository Guidelines

## Project Structure & Module Organization

- `src/alphaforge/`: Python backend package (src-layout). Major modules map to the platform layers: `data/`, `features/`, `strategy/`, `discovery/`, `backtest/`, `validation/`, plus `api/` and `cli/`.
- `tests/`: pytest suite organized by domain (e.g., `tests/test_validation/`, `tests/test_discovery/`).
- `frontend/`: Next.js (App Router) UI (`frontend/app/`, shared UI in `frontend/components/`, client utilities in `frontend/lib/`).
- Runtime artifacts: `data/` (local market-data cache), `logs/` (service logs).

## Build, Test, and Development Commands

- Full stack (recommended): `./start.sh` (backend on `:8000`, frontend on `:3000`, logs in `logs/`).
- Dev with live logs: `./start-dev.sh` (runs in a `tmux` session; may require `tmux` installed).
- Stop services: `./stop.sh`.
- Backend install: `pip install -e ".[dev]"` (or `uv venv && uv pip install -e ".[dev]"`).
- Backend run (manual): `python -m alphaforge.api.server` (after `pip install -e .`), or `PYTHONPATH=src python -m alphaforge.api.server`.
- Frontend: `cd frontend && npm install`, then `npm run dev|build|start|lint`.
- Quality gates: `ruff check src/ tests/`, `ruff format src/ tests/`, `mypy src/alphaforge/`.

## Coding Style & Naming Conventions

- Python: 4-space indentation, type hints required (mypy enforces typed defs), `ruff` line length is 100.
- Naming: modules/functions `snake_case`, classes `CamelCase`, tests `test_*.py`.
- Domain rules: avoid lookahead bias (use trailing windows only) and prefer real market data via `MarketDataLoader` (cached under `data/`).

## Testing Guidelines

- Framework: `pytest` (+ `pytest-asyncio`, `pytest-cov`).
- Run all tests: `pytest`; skip slow tests: `pytest -m "not slow"`; coverage: `pytest --cov=alphaforge`.
- Add tests alongside the relevant domain folder (e.g., validation changes → `tests/test_validation/`).

## Commit & Pull Request Guidelines

- Commits: prefer Conventional Commit-style prefixes seen in history (`feat:`, `perf:`, `refactor:`; use `fix:` when applicable) with an imperative subject.
- PRs: include a clear description, link issues, list commands run (e.g., `pytest`, `ruff`, `mypy`), and attach screenshots for UI changes.
- Don’t commit generated or local artifacts: `.venv/`, `logs/`, `data/cache/`, `frontend/.next/`, `.env*`.

## Configuration & Secrets

- Keep credentials out of Git; use environment variables (e.g., `MLFLOW_TRACKING_URI`) and document any new config in the PR description.
