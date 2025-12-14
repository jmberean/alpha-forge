# AlphaForge Context

## Project Overview

**AlphaForge** is a production-grade platform for systematic trading strategy discovery, validation, and deployment. It is designed to rigorously defend against common quantitative finance pitfalls such as overfitting, lookahead bias, and execution reality mismatches.

The system uses a defense-in-depth architecture, incorporating:
*   **Bi-temporal data architecture** to prevent lookahead bias.
*   **Genetic programming** for automated strategy discovery.
*   **Vectorized and Event-Driven backtesting** engines.
*   **Advanced statistical validation** (DSR, CPCV, SPA, Stress Testing).

## Architecture & Tech Stack

### Backend (`src/alphaforge`)
*   **Language:** Python 3.10+
*   **Frameworks:** FastAPI (API), DEAP (Genetic Programming), Optuna (Optimization).
*   **Data Analysis:** Pandas, NumPy, Polars, SciPy, PyArrow.
*   **Key Modules:**
    *   `api`: FastAPI server endpoints.
    *   `backtest`: Engines for simulation (vectorized & event-driven).
    *   `data`: Data loading, cleaning, and bi-temporal storage.
    *   `factory`: Strategy generation (Genetic Algorithms).
    *   `features`: Technical indicators and feature engineering.
    *   `strategy`: Signal generation and strategy templates.
    *   `validation`: Statistical tests (DSR, CPCV, PBO).
*   **Build System:** Hatchling (defined in `pyproject.toml`).

### Frontend (`frontend/`)
*   **Framework:** Next.js 14 (React 18).
*   **Styling:** Tailwind CSS.
*   **Components:** Framer Motion (animations), Recharts (charts).
*   **Language:** TypeScript.

## Development & Usage

### 1. Setup
*   **Backend:**
    ```bash
    source .venv/bin/activate
    uv pip install -e ".[dev]"
    ```
*   **Frontend:**
    ```bash
    cd frontend
    npm install
    ```

### 2. Running the Application
*   **API Server:**
    ```bash
    # From project root
    source .venv/bin/activate
    cd src
    python -m alphaforge.api.server
    ```
    API Docs: `http://localhost:8000/docs`

*   **Frontend:**
    ```bash
    # From frontend directory
    cd frontend
    npm run dev
    ```
    UI: `http://localhost:3000`

### 3. Testing & Verification
*   **Run Tests:**
    ```bash
    pytest
    # Specific test
    pytest tests/test_validation/test_dsr.py
    ```
*   **Linting & Formatting:**
    ```bash
    ruff check src/ tests/
    ruff format src/ tests/
    ```
*   **Type Checking:**
    ```bash
    mypy src/alphaforge/
    ```

## Key Conventions
*   **No Synthetic Data:** Tests should use real market data or realistic mocks.
*   **Type Hints:** Strict mypy compliance is required.
*   **Validation:** All features must pass lookahead bias detection.
*   **Testing:** Comprehensive test coverage for new features is mandatory.

## Current Status (MVP7)
The project has completed MVP7 (Event-Driven Backtesting) and is currently focusing on Production Infrastructure (MVP8).
*   **Completed:** Data Layer, Features, Strategy Factory, Backtesting, Validation, Optimization.
*   **Upcoming:** Kubernetes, Iceberg Data Lake, Production Execution, Monitoring.
