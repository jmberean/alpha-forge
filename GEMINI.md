# AlphaForge Context for Gemini

## Project Overview

**AlphaForge** is a production-grade platform for systematic trading strategy discovery, validation, and deployment. It implements defense-in-depth against overfitting, lookahead bias, and execution reality mismatch.

**Core Functionality:**
*   **Strategy Discovery:** Multi-objective genetic programming (NSGA-III) to evolve expression tree-based trading strategies.
*   **Validation Pipeline:** Rigorous statistical validation including Deflated Sharpe Ratio (DSR), Combinatorially Purged Cross-Validation (CPCV), Probability of Backtest Overfitting (PBO), and Hansen's SPA.
*   **Backtesting:** Both vectorized (fast) and event-driven (realistic) backtesting engines with market impact modeling.
*   **Architecture:** 5-layer architecture separating Data, Features, Strategy, Backtesting, and Validation.

**Key Technologies:**
*   **Backend:** Python 3.10+, FastAPI, NumPy, Pandas, Polars, PyArrow, yfinance, MLflow.
*   **Frontend:** Next.js 14 (TypeScript), React 18, Tailwind CSS, Framer Motion, Recharts.
*   **CLI:** Click-based command-line interface.

## Building and Running

### Development Environment
*   **Start Full Stack (Interactive):** `./start-dev.sh`
    *   Starts backend and frontend in a `tmux` session with split panes.
    *   Frontend: `http://localhost:3000`
    *   Backend: `http://localhost:8000`
    *   API Docs: `http://localhost:8000/docs`
*   **Start Full Stack (Background):** `./start.sh`
*   **Stop All Services:** `./stop.sh`

### Backend
*   **Install Dependencies:** `uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"` (or standard `pip` equivalent).
*   **Run API Server:** `python -m alphaforge.api.server`
*   **Run CLI:** `alphaforge --help`

### Frontend
*   **Directory:** `frontend/`
*   **Install Dependencies:** `npm install`
*   **Run Dev Server:** `npm run dev`

## Development Conventions

### Critical Rules
1.  **NO SYNTHETIC DATA:** All tests and development must use real market data fetched via `yfinance` (cached in `data/cache/`). Never generate random price data.
2.  **NO LOOKAHEAD BIAS:** Use trailing windows only (e.g., `rolling(20, center=False)`). Never use `.shift(-1)` or `center=True`.
3.  **Strict Typing:** Python code must be fully type-hinted and pass `mypy`.

### Code Quality & Testing
*   **Run Tests:** `pytest` (runs all tests).
    *   Specific module: `pytest tests/test_validation/ -v`
    *   With coverage: `pytest --cov=alphaforge`
*   **Linting:** `ruff check src/ tests/`
*   **Formatting:** `ruff format src/ tests/`
*   **Type Checking:** `mypy src/alphaforge/`

### Project Structure
*   `src/alphaforge/` - Python source code.
    *   `data/` - Layer 1: Data loading, caching, bi-temporal schema.
    *   `features/` - Layer 2: Technical indicators (strictly trailing).
    *   `discovery/` - Layer 3: Genetic programming and strategy evolution.
    *   `backtest/` - Layer 4: Vectorized and event-driven engines.
    *   `validation/` - Layer 5: Statistical validation pipeline.
    *   `api/` - FastAPI endpoints.
    *   `cli.py` - CLI entry point.
*   `frontend/` - Next.js application.
*   `tests/` - Pytest suite (uses real cached data).
*   `data/cache/` - Local cache for yfinance data (git-ignored).

### Key Files
*   `pyproject.toml` - Python project configuration and dependencies.
*   `CLAUDE.md` - Additional context and rules for AI assistants.
*   `README.md` - Comprehensive project documentation.
*   `start-dev.sh` - Main entry point for development.
