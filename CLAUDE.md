# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaForge is a production-grade platform for systematic trading strategy discovery, validation, and deployment. It implements defense-in-depth against overfitting, lookahead bias, and execution reality mismatch.

## Critical Rules

**NO SYNTHETIC DATA, NO PLACEHOLDERS** - All data must be real market data.
- Use yfinance for market data retrieval
- Tests use real historical data (cached for reproducibility)
- Never generate fake/random price data

## Workflow Requirements
- Update CLAUDE.md file when appropriate

## Build and Test Commands

```bash
# Install dependencies (use venv)
export PATH="$HOME/.local/bin:$PATH"
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Run tests
pytest

# Run specific test file
pytest tests/test_validation/test_dsr.py -v

# Run with coverage
pytest --cov=alphaforge --cov-report=term-missing

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/alphaforge/
```

## CLI Usage

```bash
# Load market data
alphaforge data SPY --start 2020-01-01

# Run backtest
alphaforge backtest SPY --template sma_crossover

# Validate strategy
alphaforge validate SPY --template sma_crossover --n-trials 100

# List templates
alphaforge templates
```

## Architecture

Seven-layer defense architecture:
1. **Layer 1: Bi-Temporal Data Lake** - Hot Store (Arrow/NVMe) + Cold Store (Iceberg/MinIO) with 3-timestamp schema (observation_date, release_date, transaction_time)
2. **Layer 2: Point-in-Time Feature Store** - Feast-based, ensures features use only data available at query time
3. **Layer 3: Strategy Factory** - DEAP genetic programming, Optuna optimization, Chronos-2 foundation models
4. **Layer 4: Hybrid Backtesting** - Vectorized (vectorbt) for screening, Event-driven (NautilusTrader) for validation
5. **Layer 5: Statistical Validation** - CPCV, PBO, DSR, Hansen's SPA test, stress testing
6. **Layer 6: Production Execution** - Shadow/paper/live trading with Almgren-Chriss market impact
7. **Layer 7: Governance & Monitoring** - MLflow tracking, CUSUM/SPRT monitoring, SHAP explainability

## Strategy Validation Funnel

10,000 candidates → DSR screening (0.95) → CPCV/PBO (<0.05) → Event-driven backtest → SPA + stress → Shadow trading → 2-3 deployed (~0.03% pass rate)

## Technology Stack

- **Languages**: Python 85%, Rust 10% (NautilusTrader), SQL/YAML 5%
- **Data**: Apache Iceberg, Arrow IPC, Kafka/Redpanda, PostgreSQL
- **Feature Store**: Feast, Polars, TA-Lib
- **Backtesting**: vectorbt, NautilusTrader
- **Validation**: scipy, statsmodels, arch (SPA, bootstrap)
- **Monitoring**: MLflow, Prometheus, Grafana

## Key Validation Thresholds

| Metric | Deploy | Auto-Accept |
|--------|--------|-------------|
| PBO | < 0.05 | < 0.02 |
| DSR | > 0.95 | > 0.98 |
| Sharpe | > 1.0 | > 1.5 |

## Reference

See `AlphaForge_System_Specification.md` for complete system design including data schemas, algorithms, and implementation roadmap.
