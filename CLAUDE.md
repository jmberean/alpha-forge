# CLAUDE.md

Development guidance for Claude Code when working with AlphaForge.

## Project Overview

AlphaForge is a systematic trading strategy discovery and validation platform. Defense-in-depth against overfitting, lookahead bias, and execution mismatch.

**Status**: MVP8 - Multi-objective GP discovery (NSGA-III), 300+ tests, production-ready.

## Critical Rules

### NO SYNTHETIC DATA
- All data must be real market data via yfinance
- Tests use cached real historical data
- Never generate fake/random price data
- Cache: `data/cache/` (git-ignored)

### NO UNNECESSARY MD FILES
- Never create *.md files unless explicitly requested
- Update CLAUDE.md instead of creating new docs
- Exception: PLAN.md, RESULTS.md for milestones

### CODE QUALITY
- Python 3.10+ with type hints (mypy strict)
- All code must pass: `pytest`, `ruff`, `mypy`
- Test coverage required for new features

## Quick Start

```bash
./start.sh          # Start backend + frontend (background)
./start-dev.sh      # Start with tmux split-screen
./stop.sh           # Stop all services
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Build & Test

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Test
pytest                                    # All tests
pytest tests/test_validation/ -v          # Specific module
pytest --cov=alphaforge                   # With coverage

# Quality
ruff check src/ tests/ && ruff format src/ tests/
mypy src/alphaforge/
```

## CLI

```bash
alphaforge data SPY --start 2020-01-01
alphaforge backtest SPY --template sma_crossover
alphaforge validate SPY --template sma_crossover --n-trials 100 --run-cpcv
alphaforge templates
```

## Module Structure

```
src/alphaforge/
├── data/           # Layer 1: yfinance loader, bi-temporal schema, caching
├── features/       # Layer 2: 30+ indicators, lookahead detection
├── strategy/       # Layer 3: StrategyGenome, templates
├── discovery/      # NSGA-III, expression trees, genetic operators
├── backtest/       # Layer 4: Vectorized + event-driven engines
├── validation/     # Layer 5: DSR, CPCV, PBO, SPA, stress testing
├── optimization/   # Grid, random, Optuna optimizers
├── monitoring/     # CUSUM degradation detection
├── api/            # FastAPI server
└── cli.py          # Command-line interface
```

## Validation Thresholds

| Metric | Minimum | Auto-Accept |
|--------|---------|-------------|
| Prob. of Loss | < 0.05 | < 0.02 |
| DSR | > 0.95 | > 0.98 |
| Sharpe | > 1.0 | > 1.5 |
| Stress Pass | ≥ 80% | ≥ 80% |
| SPA p-value | < 0.05 | < 0.05 |

## Common Pitfalls

- **Lookahead**: No `.shift(-1)`, no `center=True` windows
- **Data Snooping**: Always pass `n_trials` to DSR for multiple testing correction
- **Type Hints**: All functions need complete annotations
- **Test Data**: Never commit data files; use cache + .gitignore

## Plugins

### frontend-design
Use for UI/UX work in `frontend/`. Creates production-grade components.
```
/skill frontend-design
```

### ast-grep
Use for structural code search and refactoring patterns.
```
/skill ast-grep
```

### greptile
Use for deep codebase exploration and understanding complex flows.
- MCP server with list_merge_requests, get_code_review, search_custom_context tools
- Helpful for understanding multi-file interactions

## Parallel Agents

Use `run_in_background=True` for:
- Independent code reviews
- Codebase exploration across modules
- Long-running tests/builds

Stay sequential for:
- Dependent tasks
- Architecture decisions
- Security-sensitive changes

## Key Imports

```python
# Data
from alphaforge.data.loader import MarketDataLoader

# Discovery
from alphaforge.discovery import DiscoveryOrchestrator, DiscoveryConfig

# Validation
from alphaforge.validation.pipeline import ValidationPipeline

# Backtest
from alphaforge.backtest.engine import BacktestEngine

# Strategy
from alphaforge.strategy.templates import StrategyTemplates
```

See README.md for comprehensive documentation.
