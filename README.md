# AlphaForge

Production-grade platform for systematic trading strategy discovery, validation, and deployment.

## Features

- **Defense-in-depth** against overfitting, lookahead bias, and execution reality mismatch
- **Bi-temporal data architecture** with point-in-time guarantees
- **Statistical validation pipeline** with DSR, CPCV, and PBO
- **Vectorized backtesting** for high-performance strategy screening

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick Start

```python
from alphaforge import MarketDataLoader, StrategyGenome, BacktestEngine, ValidationPipeline
from alphaforge.strategy.templates import StrategyTemplates

# Load real market data
loader = MarketDataLoader()
data = loader.load("SPY", start="2020-01-01", end="2024-01-01")

# Create a strategy
strategy = StrategyTemplates.sma_crossover(fast_period=20, slow_period=50)

# Backtest
engine = BacktestEngine()
result = engine.run(strategy, data)

print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
```

## License

MIT
