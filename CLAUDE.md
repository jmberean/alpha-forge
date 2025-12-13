# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaForge is a production-grade platform for systematic trading strategy discovery, validation, and deployment. It implements defense-in-depth against overfitting, lookahead bias, and execution reality mismatch.

**Current Status**: MVP7 Complete - Full-featured systematic trading platform with bi-temporal data, 30+ advanced indicators, genetic strategy evolution, and event-driven backtesting. Production-ready with 300+ tests.

## Critical Rules

**NO SYNTHETIC DATA, NO PLACEHOLDERS** - All data must be real market data.
- Use yfinance for market data retrieval
- Tests use real historical data (cached for reproducibility)
- Never generate fake/random price data
- Data cache location: `data/cache/` (Git-ignored)

**NO UNNECESSARY MD FILES** - Do not create markdown documentation files unless explicitly requested.
- NEVER proactively create documentation files (*.md) or README files
- Only create MD files if the user explicitly requests them
- Update existing documentation (CLAUDE.md) instead of creating new files
- Exception: Required project files (PLAN.md, RESULTS.md when completing major milestones)

**Code Quality Standards**:
- Python 3.10+ required
- Type hints enforced (mypy strict mode)
- All new code must pass: pytest, ruff, mypy
- Test coverage expected for new features

## Workflow Requirements
- Update CLAUDE.md when adding new modules or architectural changes
- Document significant deviations from AlphaForge_System_Specification.md

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

# Validate strategy (full suite)
alphaforge validate SPY --template sma_crossover --n-trials 100 --run-cpcv --run-spa --run-stress

# List templates
alphaforge templates
```

## Code Architecture

### Module Organization

```
src/alphaforge/
├── data/                 # Layer 1: Data loading and storage
│   ├── loader.py        # MarketDataLoader (yfinance integration)
│   ├── schema.py        # OHLCV data schemas
│   ├── bitemporal.py    # Bi-temporal 3-timestamp schema (MVP4)
│   ├── alfred.py        # Federal Reserve vintage data (MVP4)
│   ├── universe.py      # Survivorship bias prevention (MVP4)
│   └── quality.py       # Data quality validation (MVP4)
├── features/            # Layer 2: Feature engineering
│   ├── technical.py     # Basic indicators (SMA, RSI, MACD, Bollinger, ATR, ADX)
│   ├── advanced_technical.py  # 30+ advanced indicators (MVP5)
│   ├── lookahead.py     # Lookahead bias detection (MVP5)
│   └── llm_safety.py    # LLM temporal safety framework (MVP5)
├── strategy/            # Layer 3: Strategy representation
│   ├── genome.py        # StrategyGenome (universal format)
│   └── templates.py     # Pre-built strategy templates
├── factory/             # Strategy generation (MVP6)
│   ├── genetic.py       # DEAP genetic programming
│   └── orchestrator.py  # Strategy factory coordination
├── optimization/        # Parameter optimization (MVP3)
│   ├── base.py          # Optimizer interface and ParameterSpace
│   ├── grid.py          # Grid search
│   ├── random.py        # Random search
│   └── optuna.py        # Bayesian optimization (TPE)
├── backtest/            # Layer 4: Backtesting engine
│   ├── engine.py        # Vectorized backtesting
│   ├── event_driven.py  # Event-driven backtest with queue models (MVP7)
│   ├── metrics.py       # Performance metrics
│   ├── impact.py        # Almgren-Chriss market impact (MVP2)
│   └── trades.py        # Trade analysis (MVP3)
├── validation/          # Layer 5: Statistical validation
│   ├── dsr.py           # Deflated Sharpe Ratio
│   ├── cpcv.py          # Combinatorially Purged CV
│   ├── pbo.py           # Probability of Backtest Overfitting
│   ├── spa.py           # Hansen's SPA test (MVP2)
│   ├── stress.py        # Stress testing (MVP2)
│   ├── regime.py        # Market regime detection (MVP3)
│   ├── walk_forward.py  # Walk-forward analysis (MVP3)
│   └── pipeline.py      # ValidationPipeline orchestrator
├── analysis/            # Performance analysis (MVP3)
│   └── attribution.py   # Regime-based attribution
├── monitoring/          # Production monitoring (MVP2)
│   └── cusum.py         # CUSUM degradation detection
└── cli.py               # Command-line interface
```

## Key Features by Module

**Bi-Temporal Data** (`alphaforge.data.bitemporal`): 3-timestamp schema (observation/release/transaction) for point-in-time queries, prevents lookahead bias at data layer (MVP4)

**ALFRED Integration** (`alphaforge.data.alfred`): Federal Reserve vintage data with revision tracking for macro indicators (MVP4)

**Survivorship Prevention** (`alphaforge.data.universe`): Track delisted securities, validate universe completeness (MVP4)

**Data Quality** (`alphaforge.data.quality`): Automated validation (OHLC consistency, gaps, sanity checks) (MVP4)

**Advanced Indicators** (`alphaforge.features.advanced_technical`): 30+ indicators (Stochastic, Williams %R, CCI, MFI, Ichimoku, Keltner, Donchian, Aroon, KST, etc.) (MVP5)

**Lookahead Detection** (`alphaforge.features.lookahead`): Detect centered windows, future shifts, full-sample stats in feature functions (MVP5)

**LLM Temporal Safety** (`alphaforge.features.llm_safety`): Canary questions to test LLM knowledge cutoff, prevent temporal contamination (MVP5)

**Genetic Evolution** (`alphaforge.factory.genetic`): DEAP-based genetic programming for strategy evolution (population 100, generations 50) (MVP6)

**Strategy Factory** (`alphaforge.factory.orchestrator`): Coordinate genetic evolution + template variations, generate 100+ candidates (MVP6)

**Event-Driven Backtest** (`alphaforge.backtest.event_driven`): Queue models, latency simulation (50ms), partial fills, realistic execution (MVP7)

**Implementation Shortfall** (`alphaforge.backtest.event_driven`): Compare vectorized vs event-driven results, <30% degradation threshold (MVP7)

**Optimization** (`alphaforge.optimization`): Grid/Random/Optuna optimizers with ParameterSpace (MVP3)

**Regime Detection** (`alphaforge.validation.regime`): Detect normal/trending/high_vol/crisis regimes (MVP3)

**Trade Analysis** (`alphaforge.backtest.trades`): Win rate, profit factor, expectancy (MVP3)

**Walk-Forward** (`alphaforge.validation.walk_forward`): Rolling/anchored windows with IS optimization + OOS testing (MVP3)

**Performance Attribution** (`alphaforge.analysis.attribution`): Regime-based breakdown, monthly/yearly returns (MVP3)

**SPA Test** (`alphaforge.validation.spa`): Hansen's Superior Predictive Ability test (MVP2)

**Stress Testing** (`alphaforge.validation.stress`): 3 historical + 3 synthetic scenarios (MVP2)

**CUSUM Monitoring** (`alphaforge.monitoring.cusum`): Real-time degradation detection (MVP2)

**Market Impact** (`alphaforge.backtest.impact`): Almgren-Chriss parametric model (MVP2)

## Validation Pipeline

```python
from alphaforge.validation.pipeline import ValidationPipeline

pipeline = ValidationPipeline()
result = pipeline.validate(
    strategy=my_strategy,
    data=market_data,
    n_trials=1000,        # For DSR multiple testing correction
    run_cpcv=True,        # Combinatorially Purged CV
    run_spa=True,         # Hansen's SPA test
    run_stress=True,      # Stress testing
    benchmark_returns=spy_returns,
    benchmark_name="SPY"
)

# result.passed: meets minimum thresholds
# result.auto_accept: meets strict thresholds (DSR > 0.98, Sharpe > 1.5, PBO < 0.02)
```

## Validation Thresholds

| Metric | Minimum | Auto-Accept |
|--------|---------|-------------|
| PBO | < 0.05 | < 0.02 |
| DSR | > 0.95 | > 0.98 |
| Sharpe | > 1.0 | > 1.5 |
| Stress Pass Rate | ≥ 80% | ≥ 80% |
| SPA p-value | < 0.05 | < 0.05 |

## Testing Patterns

**Use Real Market Data**:
```python
@pytest.fixture
def spy_data():
    """Real SPY data, cached for test reproducibility."""
    loader = MarketDataLoader()
    return loader.load("SPY", start="2020-01-01", end="2023-12-31")
```

**Never use synthetic price data** - Use yfinance or cached real data

## Common Pitfalls

- **Lookahead Bias**: Ensure features use only past data (no `.shift(-1)` or centered windows)
- **Data Snooping**: Always use `n_trials` parameter in DSR to account for all strategies tested
- **Test Data**: Never commit large data files; use cache and `.gitignore`
- **Type Hints**: All new functions must have complete type annotations

## Reference

- **System Specification**: `AlphaForge_System_Specification.md` - Complete system design
- **MVP Roadmap**: `MVP_ROADMAP.md` - MVP4-12 implementation plan (26-35 weeks to production)
- **README**: `README.md` - Quick start and installation
