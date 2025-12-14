# AlphaForge Quick Start Guide

Run the full AlphaForge platform with frontend + backend in 3 terminals.

## Prerequisites

- Python 3.10+ with venv
- Node.js 18+ with npm
- Port 3000 (frontend) and 8000 (backend) available

## Terminal 1: Python Backend API

```bash
# Activate Python environment
source .venv/bin/activate

# Start FastAPI server
cd src
python -m alphaforge.api.server

# Server will run on http://localhost:8000
# API docs available at http://localhost:8000/docs
```

## Terminal 2: Next.js Frontend

```bash
# Navigate to frontend
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Frontend will run on http://localhost:3000
```

## Terminal 3: Monitor / Run Commands

```bash
# Activate Python environment
source .venv/bin/activate

# Run standalone validations
alphaforge validate SPY --template sma_crossover --n-trials 100

# Or use the frontend UI at http://localhost:3000
```

## Using the Application

### Via Web UI (http://localhost:3000)

1. **View Mock Data**: Top sections show example pipeline and metrics
2. **Run Real Validation**: Scroll to "RUN VALIDATION" section at bottom
3. **Enter Symbol**: e.g., SPY, AAPL, GOOGL
4. **Select Template**: Choose from dropdown (sma_crossover, rsi_mean_reversion, etc.)
5. **Click "START VALIDATION"**: Watch real-time logs appear
6. **View Results**: Metrics appear when validation completes

### Via CLI

```bash
# Simple validation
alphaforge validate SPY --template sma_crossover

# Full validation suite
alphaforge validate SPY --template sma_crossover \
  --n-trials 1000 \
  --run-cpcv \
  --run-spa \
  --run-stress
```

### Via Python API

```python
from alphaforge import MarketDataLoader, BacktestEngine, ValidationPipeline
from alphaforge.strategy.templates import StrategyTemplates

# Load data
loader = MarketDataLoader()
data = loader.load("SPY", start="2020-01-01", end="2023-12-31")

# Get strategy
strategy = StrategyTemplates.sma_crossover()

# Run backtest
engine = BacktestEngine()
result = engine.run(strategy, data)

# Validate
pipeline = ValidationPipeline()
validation = pipeline.validate(strategy, data, n_trials=1000)

print(f"Passed: {validation.passed}")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
print(f"DSR: {validation.dsr:.4f}")
```

## API Endpoints

- `GET /` - Health check
- `POST /api/validate` - Start strategy validation
- `GET /api/validate/{id}` - Get validation results
- `GET /api/strategies` - List completed validations
- `GET /api/templates` - List available strategy templates

API docs: http://localhost:8000/docs

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Frontend (Next.js)                             │
│  http://localhost:3000                          │
│  - Dashboard UI                                 │
│  - Real-time validation runner                  │
└─────────────────┬───────────────────────────────┘
                  │ HTTP/REST
┌─────────────────▼───────────────────────────────┐
│  Backend API (FastAPI)                          │
│  http://localhost:8000                          │
│  - /api/validate - Run validations              │
│  - /api/strategies - List results               │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  AlphaForge Core (Python)                       │
│  - Data loading (yfinance)                      │
│  - Backtesting engine                           │
│  - Statistical validation                       │
│  - Strategy generation                          │
└─────────────────────────────────────────────────┘
```

## What's Mock vs Real?

### Mock Data (Example/Demo):
- Validation Pipeline visualization (10,000 → 3 funnel)
- Metrics Grid (DSR 0.96, PBO 0.04, etc.)
- Equity curve chart
- Strategy candidates list

### Real Data (Actually Running):
- "RUN VALIDATION" section at bottom of page
- Runs actual AlphaForge validation via API
- Real market data from yfinance
- Real backtest + statistical validation
- Real-time logs and results

## Example Workflow

1. **Start both servers** (Terminal 1 & 2 above)
2. **Open browser** to http://localhost:3000
3. **Scroll down** to "RUN VALIDATION"
4. **Enter SPY** and select "SMA Crossover"
5. **Click START VALIDATION**
6. **Watch logs** appear in real-time
7. **See results** (likely FAILED - SMA crossover with defaults performs poorly)
8. **Try different templates** - RSI Mean Reversion, MACD Trend, etc.

## Troubleshooting

### Backend won't start
- Check port 8000 is free: `lsof -i :8000`
- Ensure venv is activated: `which python` should show `.venv`

### Frontend won't start
- Check port 3000 is free: `lsof -i :3000`
- Delete `node_modules` and run `npm install` again

### CORS errors
- Backend should allow `http://localhost:3000` (already configured)
- Check browser console for specific errors

### Validation fails immediately
- Check backend logs in Terminal 1
- Ensure market data can be downloaded (yfinance requires internet)
- Try with cached data: run `alphaforge data SPY` first

## Next Steps

- Try different symbols: AAPL, MSFT, GOOGL
- Try different templates: each has different characteristics
- Run full validation: `--run-cpcv --run-spa --run-stress` (takes ~5-10 min)
- Build your own strategy templates in `src/alphaforge/strategy/templates.py`

## Production Deployment

This quickstart runs everything locally. For production:

1. Deploy backend to cloud (AWS Lambda, Google Cloud Run, etc.)
2. Deploy frontend to Vercel/Netlify
3. Set `NEXT_PUBLIC_API_URL` env var to backend URL
4. Add authentication (API keys, OAuth)
5. Add database for persistence (currently in-memory)
6. Scale workers for parallel validations
