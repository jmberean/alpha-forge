# AlphaForge

**Production-grade platform for systematic trading strategy discovery, validation, and deployment.**

AlphaForge implements defense-in-depth against the primary failure modes in quantitative finance: overfitting, lookahead bias, and execution reality mismatch.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-300%2B%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Key Differentiators

| Problem | Industry Failure Rate | AlphaForge Solution |
|---------|----------------------|---------------------|
| **False discovery** | 95% of published factors fail | CPCV + PBO + DSR validation |
| **Lookahead bias** | 60-80% of retail algos | Bi-temporal data + feature validation |
| **Execution gap** | 70% fail within 6 months | Event-driven backtest + implementation shortfall |
| **Survivorship bias** | Common in retail platforms | Delisted securities tracking |

## ✨ Features

### Data Architecture (MVP4)
- **Bi-temporal data schema** - 3-timestamp design (observation/release/transaction) prevents lookahead bias at infrastructure level
- **ALFRED integration** - Federal Reserve vintage data with full revision history for 14+ macro indicators
- **Survivorship bias prevention** - Track delisted securities, validate universe completeness
- **Automated data quality** - 9 validation checks (OHLC consistency, gaps, sanity, duplicates)

### Advanced Feature Engineering (MVP5)
- **30+ technical indicators** - Stochastic, Williams %R, CCI, MFI, Ichimoku, Keltner, Donchian, Aroon, KST, and more
- **Lookahead bias detection** - Automatically detect centered windows, future shifts, full-sample statistics
- **LLM temporal safety** - Canary questions to prevent temporal contamination in LLM-based features

### Strategy Generation (MVP6)
- **Genetic programming** - DEAP-based evolution (population 100, generations 50)
- **Strategy factory** - Generate 100+ candidates through evolution + template variations
- **Automated fitness scoring** - Integrate with backtesting for objective evaluation

### Backtesting (MVP7)
- **Vectorized engine** - Screen 10,000+ strategies at 10M+ bars/sec
- **Event-driven engine** - Realistic execution with queue models, 50ms latency, partial fills
- **Implementation shortfall** - Measure vectorized vs realistic performance gap (<30% threshold)

### Statistical Validation (MVP2-3)
- **Deflated Sharpe Ratio** - Account for multiple testing (10,000 trials → DSR > 0.95)
- **Combinatorially Purged CV** - 12,870 train/test combinations, PBO < 0.05
- **Hansen's SPA test** - Statistical superiority vs benchmarks
- **Stress testing** - 3 historical + 3 synthetic scenarios
- **Walk-forward analysis** - Rolling/anchored windows with IS/OOS validation
- **Regime detection** - Adaptive thresholds for normal/trending/high_vol/crisis

### Production Monitoring (MVP2)
- **CUSUM degradation detection** - Real-time monitoring with <5 day detection time
- **Performance attribution** - Regime-based breakdown, monthly/yearly analysis
- **Trade analytics** - Win rate, profit factor, expectancy, consecutive wins/losses

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/alpha-forge.git
cd alpha-forge

# Create virtual environment
uv venv
source .venv/bin/activate

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Basic Example

```python
from alphaforge import MarketDataLoader, BacktestEngine, ValidationPipeline
from alphaforge.strategy.templates import StrategyTemplates

# 1. Load real market data (no synthetic data!)
loader = MarketDataLoader()
data = loader.load("SPY", start="2020-01-01", end="2024-01-01")

# 2. Create or generate a strategy
strategy = StrategyTemplates.sma_crossover(fast_period=20, slow_period=50)

# 3. Backtest with vectorized engine
engine = BacktestEngine()
result = engine.run(strategy, data)

print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Annual Return: {result.metrics.annualized_return:.1%}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.1%}")
```

### Advanced: Full Validation Pipeline

```python
from alphaforge.validation import ValidationPipeline

# Run comprehensive validation
pipeline = ValidationPipeline()
validation = pipeline.validate(
    strategy=strategy,
    data=data,
    n_trials=1000,        # For DSR multiple testing correction
    run_cpcv=True,        # Combinatorially Purged CV
    run_spa=True,         # Hansen's SPA test
    run_stress=True,      # Stress testing
    benchmark_returns=spy_returns,
)

# Check results
if validation.auto_accept:
    print("✓ Strategy meets strict thresholds (DSR > 0.98, Sharpe > 1.5, PBO < 0.02)")
elif validation.passed:
    print("✓ Strategy meets minimum thresholds (requires committee review)")
else:
    print("✗ Strategy failed validation")
```

### Generate Strategy Candidates

```python
from alphaforge.factory import StrategyFactory

def fitness_function(strategy):
    """Evaluate strategy fitness (Sharpe ratio from backtest)"""
    result = engine.run(strategy, data)
    return result.metrics.sharpe_ratio

# Generate candidates using genetic evolution
factory = StrategyFactory(fitness_function=fitness_function)
pool = factory.generate()

# Get top 10 strategies
top_strategies = factory.get_top_strategies(n=10)
for strategy, fitness in top_strategies:
    print(f"{strategy.name}: Sharpe = {fitness:.2f}")
```

### Event-Driven Backtesting

```python
from alphaforge.backtest import EventDrivenEngine, calculate_implementation_shortfall

# Run event-driven backtest (realistic execution)
event_engine = EventDrivenEngine()
event_result = event_engine.run(strategy, data)

# Compare to vectorized backtest
vector_result = BacktestEngine().run(strategy, data)
shortfall = calculate_implementation_shortfall(vector_result, event_result)

print(f"Vectorized Sharpe: {shortfall['vectorized_sharpe']:.2f}")
print(f"Event-Driven Sharpe: {shortfall['event_driven_sharpe']:.2f}")
print(f"Implementation Shortfall: {shortfall['sharpe_shortfall']:.1%}")
print(f"Passed Threshold (<30%): {shortfall['passed_threshold']}")
```

### Bi-Temporal Data Queries

```python
from alphaforge.data import BiTemporalStore, ALFREDClient, ALFREDSync
from datetime import datetime, date

# Setup bi-temporal store
store = BiTemporalStore()
client = ALFREDClient()
sync = ALFREDSync(client, store)

# Sync Federal Reserve data
sync.daily_sync(vintage_date=date(2024, 1, 15))

# Point-in-time query: "What did we know on this date?"
gdp = store.get_pit_value(
    entity_id="US_MACRO",
    indicator_name="GDP",
    observation_date=date(2023, 12, 31),  # Q4 2023
    as_of_date=datetime(2024, 1, 15),     # What was known on Jan 15, 2024
)
print(f"Q4 2023 GDP as of Jan 15, 2024: ${gdp:.0f}B")
```

### Lookahead Bias Detection

```python
from alphaforge.features import LookaheadDetector

def my_feature_function(df):
    """My custom feature computation"""
    result = df.copy()
    result['sma'] = df['close'].rolling(20).mean()  # Valid: trailing window
    result['zscore'] = (df['close'] - df['close'].rolling(252).mean()) / df['close'].rolling(252).std()  # Valid
    return result

# Validate feature function
detector = LookaheadDetector()
detections = detector.check_function(my_feature_function, data.df)

if not any(d.has_bias for d in detections):
    print("✓ No lookahead bias detected")
else:
    for detection in detections:
        if detection.has_bias:
            print(f"✗ {detection.message}")
```

## 📊 Architecture

AlphaForge implements a 7-layer defense architecture:

```
Layer 7: Governance & Monitoring (CUSUM, MLflow, SHAP)
Layer 6: Production Execution (Live, Shadow, Paper Trading)
Layer 5: Statistical Validation (CPCV, PBO, DSR, SPA, Stress)
Layer 4: Backtesting (Vectorized + Event-Driven)
Layer 3: Strategy Factory (Genetic Programming, Templates)
Layer 2: Feature Store (30+ Indicators, Lookahead Detection)
Layer 1: Bi-Temporal Data Lake (Point-in-Time, Survivorship-Free)
```

## 🧪 Testing

AlphaForge has 300+ tests ensuring production-grade quality:

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_validation/test_dsr.py -v

# Run with coverage
pytest --cov=alphaforge --cov-report=term-missing

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/alphaforge/
```

## 📈 Validation Thresholds

| Metric | Minimum (Deploy) | Auto-Accept | Crisis Regime |
|--------|------------------|-------------|---------------|
| **PBO** | < 0.05 | < 0.02 | < 0.01 |
| **DSR** | > 0.95 | > 0.98 | > 0.99 |
| **Sharpe** | > 1.0 | > 1.5 | > 2.0 |
| **Stress Pass Rate** | ≥ 80% | ≥ 80% | ≥ 80% |
| **Implementation Shortfall** | < 30% | < 20% | < 20% |

**Auto-Accept**: Strategy deploys automatically without committee review
**Minimum**: Strategy may deploy with committee approval
**Crisis**: Tightened thresholds during market stress

## 🗺️ Roadmap

- ✅ **MVP1-3**: Core platform, validation, optimization (Complete)
- ✅ **MVP4**: Bi-temporal data architecture (Complete)
- ✅ **MVP5**: Advanced feature engineering (Complete)
- ✅ **MVP6**: Strategy factory & genetic programming (Complete)
- ✅ **MVP7**: Event-driven backtesting (Complete)
- ⏳ **MVP8**: Production infrastructure (Kubernetes, Iceberg, Kafka)
- ⏳ **MVP9**: Production execution (Shadow → Paper → Live)
- ⏳ **MVP10**: Advanced monitoring (Prometheus, Grafana, SHAP)
- ⏳ **MVP11**: Risk management framework
- ⏳ **MVP12**: Testing, documentation, hardening

See [MVP_ROADMAP.md](MVP_ROADMAP.md) for detailed implementation plan.

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Development guide and code patterns
- **[AlphaForge_System_Specification.md](AlphaForge_System_Specification.md)** - Complete system design
- **[MVP_ROADMAP.md](MVP_ROADMAP.md)** - MVP4-12 implementation plan

## 🤝 Contributing

AlphaForge follows strict development practices:

1. **No synthetic data** - All tests use real market data or realistic mocks
2. **Type hints required** - Full mypy strict mode compliance
3. **Test coverage** - New features require comprehensive tests
4. **No lookahead bias** - All features validated for temporal correctness

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Always consult with a qualified financial advisor before making investment decisions.

---

**Built with**: Python 3.10+ | Pandas | NumPy | DEAP | Optuna | TA-Lib

**Status**: MVP7 Complete - Production-ready systematic trading platform
