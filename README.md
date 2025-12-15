# AlphaForge

**Production-grade platform for systematic trading strategy discovery, validation, and deployment.**

AlphaForge implements defense-in-depth against the primary failure modes in quantitative finance: **overfitting**, **lookahead bias**, and **execution reality mismatch**. It combines multi-objective genetic programming with rigorous statistical validation to discover trading strategies that are robust, interpretable, and production-ready.

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Strategy Discovery System](#strategy-discovery-system)
- [Validation Pipeline](#validation-pipeline)
- [Backtesting Engine](#backtesting-engine)
- [Technical Indicators](#technical-indicators)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Key Features

### Multi-Objective Strategy Discovery (NSGA-III)

- **Expression tree genetic programming** evolves mathematical formulas representing trading signals
- **4 simultaneous objectives**: Sharpe ratio, maximum drawdown, turnover, and complexity
- **Pareto front optimization** finds non-dominated strategies across all objectives
- **Factor zoo** maintains a library of validated, unique formulas
- **35+ typed operators** across temporal, cross-sectional, arithmetic, and logical categories

### Defense-in-Depth Validation

| Problem | Industry Failure Rate | AlphaForge Solution |
|---------|----------------------|---------------------|
| False discovery (overfitting) | 95% of published factors | CPCV + PBO + DSR validation |
| Lookahead bias | 60-80% of retail algos | Bi-temporal data architecture |
| Execution gap | 70% fail within 6 months | Event-driven backtesting + impact modeling |

### Statistical Validation Pipeline

- **Deflated Sharpe Ratio (DSR)**: Multiple testing correction accounting for all strategies tested
- **Combinatorially Purged Cross-Validation (CPCV)**: Tests 12,870 train/test combinations with temporal embargo
- **Probability of Backtest Overfitting (PBO)**: Quantifies likelihood strategy is overfit
- **Hansen's Superior Predictive Ability (SPA)**: Tests statistical significance vs benchmark
- **Stress Testing**: 3 historical scenarios + 3 synthetic shocks

### Real Market Data Only

- **No synthetic data, no placeholders** - all data from yfinance
- **Parquet caching** with automatic expiry and metadata tracking
- **Bi-temporal awareness** tracks observation, release, and transaction timestamps

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/alphaforge/alpha-forge
cd alpha-forge

# Start the full stack (backend + frontend)
./start.sh

# Open in browser
open http://localhost:3000
```

The `start.sh` script:
- Creates a Python virtual environment if needed
- Installs all dependencies
- Starts the FastAPI backend on port 8000
- Starts the Next.js frontend on port 3000
- Logs to `logs/backend.log` and `logs/frontend.log`

For interactive development with split-screen logs:
```bash
./start-dev.sh  # Uses tmux
```

Stop all services:
```bash
./stop.sh
```

---

## Architecture

AlphaForge uses a **5-layer architecture** with strict separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: VALIDATION PIPELINE                                │
│ DSR | CPCV | PBO | Hansen's SPA | Stress Testing           │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: BACKTESTING ENGINE                                 │
│ Vectorized (numpy) | Event-Driven | Market Impact Model    │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: STRATEGY DISCOVERY                                 │
│ Expression Trees | NSGA-III | Genetic Operators            │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: FEATURE ENGINEERING                                │
│ Technical Indicators | Lookahead Detection | PIT Safety    │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: DATA LAYER                                         │
│ MarketDataLoader | Parquet Cache | Bi-Temporal Schema      │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend (Python 3.10+)**
- FastAPI + Uvicorn for async REST API
- NumPy + Pandas for vectorized computation
- Polars for high-performance data operations
- PyArrow for Parquet serialization
- yfinance for market data
- MLflow for experiment tracking

**Frontend (TypeScript)**
- Next.js 14 with App Router
- React 18 with Hooks
- Tailwind CSS for styling
- Framer Motion for animations
- Recharts for data visualization

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher (for frontend)
- Git

### Backend Installation

```bash
# Using uv (recommended)
export PATH="$HOME/.local/bin:$PATH"
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Frontend Installation

```bash
cd frontend
npm install
```

### Verify Installation

```bash
# Run tests
pytest

# Check types
mypy src/alphaforge/

# Lint
ruff check src/ tests/
```

---

## Usage

### Web Interface

The web interface provides a terminal-inspired dashboard for:

1. **Strategy Discovery**: Run NSGA-III evolution with configurable objectives
2. **Genetic Factory**: Template-based genetic optimization
3. **Validation Runner**: Validate individual strategies
4. **Pipeline Statistics**: View accumulated validation metrics
5. **Strategy List**: Browse all validated strategies

Access at `http://localhost:3000` after running `./start.sh`.

### Command Line Interface

```bash
# Load market data
alphaforge data SPY --start 2020-01-01

# Run backtest with a template strategy
alphaforge backtest SPY --template sma_crossover

# Full validation suite
alphaforge validate SPY --template sma_crossover \
    --n-trials 100 \
    --run-cpcv \
    --run-spa \
    --run-stress

# List available strategy templates
alphaforge templates
```

### Python API

#### Loading Market Data

```python
from alphaforge.data.loader import MarketDataLoader

loader = MarketDataLoader()
data = loader.load("SPY", start="2020-01-01", end="2023-12-31")

print(f"Loaded {len(data.df)} trading days")
print(f"Columns: {list(data.df.columns)}")
```

#### Running a Backtest

```python
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.strategy.templates import StrategyTemplates
from alphaforge.data.loader import MarketDataLoader

# Load data
loader = MarketDataLoader()
data = loader.load("SPY", start="2020-01-01", end="2023-12-31")

# Get strategy template
strategy = StrategyTemplates.sma_crossover()

# Run backtest
engine = BacktestEngine(initial_capital=100000)
result = engine.run(strategy, data)

# View results
print(result.summary())
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.1%}")
print(f"Total Return: {result.metrics.total_return:.1%}")
```

#### Full Validation Pipeline

```python
from alphaforge.validation.pipeline import ValidationPipeline
from alphaforge.strategy.templates import StrategyTemplates
from alphaforge.data.loader import MarketDataLoader

# Load data
loader = MarketDataLoader()
data = loader.load("SPY", start="2020-01-01", end="2023-12-31")
spy_benchmark = loader.load("SPY", start="2020-01-01", end="2023-12-31")

# Create strategy
strategy = StrategyTemplates.rsi_mean_reversion()

# Run full validation
pipeline = ValidationPipeline()
result = pipeline.validate(
    strategy=strategy,
    data=data,
    n_trials=100,           # Account for multiple testing
    run_cpcv=True,          # Combinatorially Purged CV
    run_spa=True,           # Hansen's SPA test
    run_stress=True,        # Stress testing
    benchmark_returns=spy_benchmark.df['close'].pct_change(),
    benchmark_name="SPY",
)

# Check results
print(result.summary())
print(f"Passed: {result.passed}")
print(f"Auto-Accept: {result.auto_accept}")
```

---

## Strategy Discovery System

The discovery system uses **multi-objective genetic programming** to evolve trading strategies represented as expression trees.

### Running Discovery

```python
from alphaforge.discovery import DiscoveryOrchestrator, DiscoveryConfig
from alphaforge.data.loader import MarketDataLoader

# Load data
loader = MarketDataLoader()
data = loader.load("SPY", start="2020-01-01", end="2023-12-31")

# Configure discovery
config = DiscoveryConfig(
    population_size=200,      # Population per generation
    n_generations=100,        # Evolution iterations
    n_objectives=4,           # Sharpe, MaxDD, Turnover, Complexity
    min_sharpe=0.5,           # Minimum threshold for validation
    max_turnover=0.2,         # Maximum daily turnover
    max_complexity=0.7,       # Maximum complexity score
    validation_split=0.3,     # 30% held out for validation
    seed=42,                  # Reproducibility
)

# Run discovery
orchestrator = DiscoveryOrchestrator(data.df, config)
result = orchestrator.discover()

# Access results
print(f"Pareto front size: {len(result.pareto_front)}")
print(f"Factor zoo size: {len(result.factor_zoo)}")
print(f"Generations completed: {result.n_generations}")

# Best by each objective
for obj, ind in result.best_by_objective.items():
    print(f"Best {obj}: {ind.tree.formula}")
    print(f"  Fitness: {ind.fitness}")
```

### Expression Tree Operators

The system includes **35+ strongly-typed operators**:

**Temporal Operators** (trailing windows only - no lookahead):
| Operator | Description | Example |
|----------|-------------|---------|
| `delay(x, d)` | Lag series by d periods | `delay(close, 5)` |
| `ts_mean(x, w)` | Rolling mean | `ts_mean(close, 20)` |
| `ts_std(x, w)` | Rolling standard deviation | `ts_std(returns, 20)` |
| `ts_rank(x, w)` | Rolling percentile rank | `ts_rank(close, 10)` |
| `ts_corr(x, y, w)` | Rolling correlation | `ts_corr(close, volume, 20)` |
| `ts_cov(x, y, w)` | Rolling covariance | `ts_cov(close, volume, 20)` |
| `ts_min(x, w)` | Rolling minimum | `ts_min(low, 20)` |
| `ts_max(x, w)` | Rolling maximum | `ts_max(high, 20)` |
| `ts_argmax(x, w)` | Days since maximum | `ts_argmax(close, 20)` |
| `ts_argmin(x, w)` | Days since minimum | `ts_argmin(close, 20)` |

**Cross-Sectional Operators**:
| Operator | Description |
|----------|-------------|
| `rank(x)` | Percentile rank |
| `scale(x)` | Normalize to sum to 1 |
| `zscore(x)` | Standardize (mean=0, std=1) |

**Arithmetic Operators**:
| Operator | Description |
|----------|-------------|
| `add`, `sub`, `mul`, `div` | Binary operations |
| `abs`, `neg`, `log`, `sqrt`, `sign` | Unary operations |

**Logical Operators**:
| Operator | Description |
|----------|-------------|
| `gt`, `lt`, `gte`, `lte`, `eq` | Comparisons |
| `and_`, `or_`, `not_` | Boolean logic |
| `if_else` | Conditional |

### Example Discovered Formulas

```python
# Momentum with volatility adjustment
rank(ts_mean(close, 20) / ts_std(close, 20))

# Mean reversion signal
rank(delay(close, 1) - ts_mean(close, 10))

# Volume-price correlation
ts_corr(rank(close), rank(volume), 20)

# Multi-factor combination
mul(rank(ts_mean(close, 50)), neg(ts_std(close, 20)))
```

---

## Validation Pipeline

### Validation Stages

1. **Initial Backtest**: Calculate basic performance metrics
2. **DSR Screening**: Multiple testing correction for all strategies tested
3. **CPCV Validation**: 12,870 train/test combinations with temporal purging
4. **PBO Calculation**: Probability that strategy is overfit
5. **SPA Test** (optional): Statistical significance vs benchmark
6. **Stress Testing** (optional): Historical and synthetic scenarios

### Validation Thresholds

| Metric | Minimum (Deploy) | Auto-Accept |
|--------|------------------|-------------|
| PBO | < 0.05 | < 0.02 |
| DSR | > 0.95 | > 0.98 |
| Sharpe | > 1.0 | > 1.5 |
| Stress Pass Rate | ≥ 80% | ≥ 80% |
| SPA p-value | < 0.05 | < 0.05 |

### Stress Scenarios

**Historical Replays**:
- 2008 Financial Crisis (Sep-Dec 2008)
- 2020 COVID Crash (Feb-Mar 2020)
- 2022 Rate Hikes (Jan-Oct 2022)

**Synthetic Shocks**:
- Correlation Spike (all assets correlate at 0.95)
- Volatility 3x (triple normal volatility)
- Liquidity Drain (5x spreads, 80% less depth)

---

## Backtesting Engine

AlphaForge provides two backtesting modes:

### Vectorized Backtesting (Fast)

```python
from alphaforge.backtest.engine import BacktestEngine

engine = BacktestEngine(
    initial_capital=100000,
    commission_pct=0.001,   # 0.1% commission
    slippage_pct=0.0005,    # 0.05% slippage
)

result = engine.run(strategy, data)
```

- **Performance**: Millions of bars per second
- **Use case**: Screening large numbers of strategies
- **Trade-off**: No detailed trade tracking

### Event-Driven Backtesting (Realistic)

```python
from alphaforge.backtest.event_driven import EventDrivenEngine

engine = EventDrivenEngine(
    initial_capital=100000,
    latency_ms=50,           # Simulated execution latency
    partial_fills=True,      # Allow partial order fills
)

result = engine.run(strategy, data)
```

- **Features**: Queue position modeling, partial fills, market impact
- **Use case**: Final validation before deployment
- **Metric**: Implementation shortfall < 30%

### Market Impact Model (Almgren-Chriss)

```python
Total Impact = Permanent + Temporary

Permanent = 0.314 * sqrt(volatility) * (order_size / daily_volume)
Temporary = 0.142 * sqrt(volatility) * (order_size / daily_volume)^0.6 / time_horizon
```

---

## Technical Indicators

All indicators use **trailing windows only** (`center=False`) to prevent lookahead bias.

### Available Indicators

```python
from alphaforge.features.technical import TechnicalIndicators

# Compute all indicators at once
df = TechnicalIndicators.compute_all(data.df)

# Or compute individually
sma_20 = TechnicalIndicators.sma(df['close'], period=20)
rsi_14 = TechnicalIndicators.rsi(df['close'], period=14)
macd, signal, hist = TechnicalIndicators.macd(df['close'])
upper, middle, lower = TechnicalIndicators.bollinger_bands(df['close'])
```

| Category | Indicators |
|----------|------------|
| **Trend** | SMA, EMA, MACD |
| **Momentum** | RSI, Stochastic, Williams %R, Momentum |
| **Volatility** | ATR, Bollinger Bands, Rolling Volatility |
| **Trend Strength** | ADX |
| **Volume** | OBV, Volume SMA Ratio, VWAP |

### Point-in-Time Safety

```python
# FORBIDDEN: Full-sample statistics (lookahead)
df['z_score'] = (df['price'] - df['price'].mean()) / df['price'].std()

# REQUIRED: Rolling/expanding only
df['z_score'] = df['price'].rolling(252).apply(
    lambda x: (x.iloc[-1] - x.mean()) / x.std()
)

# FORBIDDEN: Centered windows (lookahead)
df['ma'] = df['price'].rolling(20, center=True).mean()

# REQUIRED: Trailing windows only
df['ma'] = df['price'].rolling(20, center=False).mean()
```

---

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/validate` | POST | Start strategy validation |
| `/api/validate/{id}` | GET | Get validation result |
| `/api/factory` | POST | Start strategy factory |
| `/api/factory/{id}` | GET | Get factory result |
| `/api/discovery` | POST | Start NSGA-III discovery |
| `/api/discovery/{id}` | GET | Get discovery result |
| `/api/strategies` | GET | List all validated strategies |
| `/api/templates` | GET | List strategy templates |
| `/api/metrics/latest` | GET | Get latest validation metrics |
| `/api/pipeline-stats` | GET | Get pipeline statistics |
| `/api/system/status` | GET | Get system status |

### Example API Calls

```bash
# Start a validation
curl -X POST http://localhost:8000/api/validate \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "template": "sma_crossover",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "n_trials": 100,
    "run_cpcv": true
  }'

# Start discovery
curl -X POST http://localhost:8000/api/discovery \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "population_size": 100,
    "n_generations": 20,
    "n_objectives": 4
  }'

# Get pipeline stats
curl http://localhost:8000/api/pipeline-stats
```

---

## Configuration

### ValidationThresholds

```python
from alphaforge.validation.pipeline import ValidationThresholds

thresholds = ValidationThresholds(
    # DSR thresholds
    dsr_confidence=0.95,           # Minimum DSR for passing

    # PBO thresholds
    pbo_deploy=0.05,               # Maximum PBO to deploy
    pbo_auto_accept=0.02,          # Maximum PBO for auto-accept

    # Sharpe thresholds
    min_sharpe=1.0,                # Minimum Sharpe for deployment
    min_sharpe_auto=1.5,           # Minimum for auto-accept

    # SPA test
    spa_pvalue_threshold=0.05,     # P-value threshold
    spa_bootstrap_reps=1000,       # Bootstrap replications

    # Stress testing
    stress_pass_rate=0.80,         # Minimum pass rate (80%)
    stress_min_sharpe=0.0,         # Minimum Sharpe during stress
    stress_max_drawdown=0.50,      # Maximum drawdown during stress

    # CPCV
    cpcv_n_splits=16,              # Number of time splits
    cpcv_test_splits=8,            # Test splits per combination
    cpcv_embargo_pct=0.02,         # Embargo between train/test
    cpcv_max_combinations=1000,    # Limit for speed
)
```

### DiscoveryConfig

```python
from alphaforge.discovery import DiscoveryConfig

config = DiscoveryConfig(
    population_size=200,           # Individuals per generation
    n_generations=100,             # Evolution iterations
    n_objectives=4,                # Optimization objectives
    crossover_prob=0.9,            # Crossover probability
    mutation_prob=0.3,             # Mutation probability
    diversity_injection=20,        # Inject diversity every N gens
    min_sharpe=0.5,                # Minimum Sharpe threshold
    max_turnover=0.2,              # Maximum turnover
    max_complexity=0.7,            # Maximum complexity score
    validation_split=0.3,          # Validation data fraction
    seed=None,                     # Random seed
)
```

---

## Testing

AlphaForge has **300+ tests** covering all major components.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=alphaforge --cov-report=term-missing

# Run specific module
pytest tests/test_validation/ -v

# Run specific test file
pytest tests/test_discovery/test_nsga3.py -v

# Skip slow tests
pytest -m "not slow"

# Run integration tests only
pytest -m integration
```

### Test Fixtures

All tests use **real market data** (cached for reproducibility):

```python
@pytest.fixture(scope="session")
def spy_data():
    """Real SPY data from 2020-2023."""
    loader = MarketDataLoader()
    return loader.load("SPY", start="2020-01-01", end="2023-12-31")
```

---

## Project Structure

```
alpha-forge/
├── src/alphaforge/
│   ├── data/                    # Layer 1: Data loading
│   │   ├── loader.py            # MarketDataLoader (yfinance + cache)
│   │   ├── schema.py            # OHLCVData schema
│   │   ├── bitemporal.py        # Bi-temporal 3-timestamp schema
│   │   ├── alfred.py            # Federal Reserve vintage data
│   │   ├── universe.py          # Survivorship bias prevention
│   │   └── quality.py           # Data quality validation
│   │
│   ├── features/                # Layer 2: Feature engineering
│   │   ├── technical.py         # 15+ technical indicators
│   │   ├── advanced_technical.py # 30+ advanced indicators
│   │   ├── lookahead.py         # Lookahead bias detection
│   │   └── llm_safety.py        # LLM temporal safety
│   │
│   ├── strategy/                # Strategy representation
│   │   ├── genome.py            # StrategyGenome
│   │   ├── templates.py         # Pre-built templates
│   │   └── signals.py           # Signal generation
│   │
│   ├── discovery/               # Layer 3: Strategy discovery
│   │   ├── expression/          # Expression trees
│   │   │   ├── tree.py          # ExpressionTree class
│   │   │   ├── nodes.py         # AST nodes
│   │   │   ├── types.py         # Operator signatures
│   │   │   └── compiler.py      # Compile to pandas
│   │   ├── operators/           # Genetic operators
│   │   │   ├── crossover.py     # Subtree/uniform crossover
│   │   │   ├── mutation.py      # 4 mutation types
│   │   │   └── selection.py     # Tournament/Pareto selection
│   │   ├── evolution/           # Evolution engine
│   │   │   └── nsga3.py         # NSGA-III optimizer
│   │   └── orchestrator.py      # Main discovery interface
│   │
│   ├── backtest/                # Layer 4: Backtesting
│   │   ├── engine.py            # Vectorized backtesting
│   │   ├── event_driven.py      # Event-driven backtest
│   │   ├── metrics.py           # Performance metrics
│   │   ├── portfolio.py         # Portfolio management
│   │   ├── impact.py            # Almgren-Chriss impact
│   │   └── trades.py            # Trade analysis
│   │
│   ├── validation/              # Layer 5: Statistical validation
│   │   ├── pipeline.py          # ValidationPipeline
│   │   ├── dsr.py               # Deflated Sharpe Ratio
│   │   ├── cpcv.py              # Combinatorially Purged CV
│   │   ├── pbo.py               # Probability of Overfitting
│   │   ├── spa.py               # Hansen's SPA test
│   │   ├── stress.py            # Stress testing
│   │   ├── regime.py            # Regime detection
│   │   └── walk_forward.py      # Walk-forward analysis
│   │
│   ├── optimization/            # Parameter optimization
│   │   ├── grid.py              # Grid search
│   │   ├── random.py            # Random search
│   │   └── optuna.py            # Bayesian optimization
│   │
│   ├── monitoring/              # Production monitoring
│   │   └── cusum.py             # CUSUM degradation detection
│   │
│   ├── api/                     # REST API
│   │   └── server.py            # FastAPI application
│   │
│   └── cli.py                   # Command-line interface
│
├── frontend/                    # Next.js frontend
│   ├── app/                     # App router pages
│   ├── components/              # React components
│   │   ├── StrategyDiscovery.tsx
│   │   ├── ParetoFrontViz.tsx
│   │   ├── FactorZoo.tsx
│   │   ├── StrategyFactory.tsx
│   │   ├── ValidationRunner.tsx
│   │   └── ...
│   └── lib/
│       └── api.ts               # Typed API client
│
├── tests/                       # Test suite (300+ tests)
│   ├── test_data/
│   ├── test_features/
│   ├── test_discovery/
│   ├── test_backtest/
│   ├── test_validation/
│   └── conftest.py              # Shared fixtures
│
├── start.sh                     # Start full stack (background)
├── start-dev.sh                 # Start full stack (tmux)
├── stop.sh                      # Stop all services
├── pyproject.toml               # Python project config
└── CLAUDE.md                    # Development guidelines
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure all tests pass: `pytest`
4. Ensure code quality: `ruff check src/ tests/ && mypy src/alphaforge/`
5. Submit a pull request

### Code Standards

- Python 3.10+ with type hints (mypy strict mode)
- No synthetic/fake data - use real market data via yfinance
- All indicators use trailing windows only (no lookahead)
- Test coverage expected for new features
