"""
FastAPI server for AlphaForge.

Provides REST API endpoints to run validations and retrieve results.
"""

import asyncio
from datetime import datetime
from typing import Optional
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from alphaforge.backtest.engine import BacktestEngine
from alphaforge.data.loader import MarketDataLoader
from alphaforge.strategy.templates import StrategyTemplates
from alphaforge.validation.pipeline import ValidationPipeline

# Create FastAPI app
app = FastAPI(
    title="AlphaForge API",
    description="Production-grade systematic trading strategy validation",
    version="0.1.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for validation results
validation_results = {}
validation_status = {}


# Request/Response Models
class ValidationRequest(BaseModel):
    symbol: str
    template: str
    start_date: Optional[str] = "2020-01-01"
    end_date: Optional[str] = "2023-12-31"
    n_trials: int = 100
    run_cpcv: bool = False
    run_spa: bool = False
    run_stress: bool = False


class ValidationResponse(BaseModel):
    validation_id: str
    status: str
    message: str


class ValidationResult(BaseModel):
    validation_id: str
    status: str
    strategy_name: str
    passed: bool
    metrics: dict
    timestamp: str
    logs: list[str]


class StrategyInfo(BaseModel):
    id: str
    name: str
    type: str
    status: str
    sharpe: float
    dsr: float
    annual_return: float


# Helper function to run validation
async def run_validation_task(validation_id: str, request: ValidationRequest):
    """Background task to run validation."""
    logs = []

    try:
        # Update status
        validation_status[validation_id] = "running"
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting validation for {request.symbol}")

        # Load data
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading market data...")
        loader = MarketDataLoader()
        data = loader.load(request.symbol, start=request.start_date, end=request.end_date)
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Loaded {len(data.df)} trading days")

        # Get strategy
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Creating strategy from template...")
        strategy_func = getattr(StrategyTemplates, request.template, None)
        if not strategy_func:
            raise ValueError(f"Unknown template: {request.template}")
        strategy = strategy_func()
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Strategy: {strategy.name}")

        # Run backtest
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Running backtest...")
        engine = BacktestEngine()
        backtest_result = engine.run(strategy, data)
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Sharpe: {backtest_result.metrics.sharpe_ratio:.2f}")

        # Run validation
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Running statistical validation...")
        pipeline = ValidationPipeline()

        # Get benchmark if SPA requested
        benchmark_returns = None
        if request.run_spa:
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading SPY benchmark...")
            spy_data = loader.load("SPY", start=request.start_date, end=request.end_date)
            spy_result = engine.run(StrategyTemplates.buy_and_hold(), spy_data)
            benchmark_returns = spy_result.returns

        validation_result = pipeline.validate(
            strategy=strategy,
            data=data,
            n_trials=request.n_trials,
            run_cpcv=request.run_cpcv,
            run_spa=request.run_spa,
            run_stress=request.run_stress,
            benchmark_returns=benchmark_returns,
            benchmark_name="SPY" if request.run_spa else None,
        )

        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ━━━ VALIDATION COMPLETE ━━━")
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {'✓ PASSED' if validation_result.passed else '✗ FAILED'}")

        # Store result
        validation_results[validation_id] = {
            "validation_id": validation_id,
            "status": "completed",
            "strategy_name": strategy.name,
            "passed": validation_result.passed,
            "metrics": {
                "sharpe_ratio": backtest_result.metrics.sharpe_ratio,
                "dsr": validation_result.dsr_result.dsr_pvalue if validation_result.dsr_result else 0.0,
                "pbo": validation_result.pbo_result.pbo if validation_result.pbo_result else None,
                "annual_return": backtest_result.metrics.annualized_return,
                "max_drawdown": backtest_result.metrics.max_drawdown,
                "sortino_ratio": backtest_result.metrics.sortino_ratio,
                "total_return": backtest_result.metrics.total_return,
                "volatility": backtest_result.metrics.volatility,
            },
            "timestamp": datetime.now().isoformat(),
            "logs": logs,
        }
        validation_status[validation_id] = "completed"

    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Error: {str(e)}")
        validation_results[validation_id] = {
            "validation_id": validation_id,
            "status": "failed",
            "strategy_name": request.template,
            "passed": False,
            "metrics": {},
            "timestamp": datetime.now().isoformat(),
            "logs": logs,
        }
        validation_status[validation_id] = "failed"


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "AlphaForge API",
        "version": "0.1.0",
        "status": "online",
    }


@app.post("/api/validate", response_model=ValidationResponse)
async def validate_strategy(request: ValidationRequest, background_tasks: BackgroundTasks):
    """
    Start a strategy validation.

    Returns immediately with a validation_id. Use GET /api/validate/{validation_id}
    to poll for results.
    """
    validation_id = str(uuid.uuid4())

    # Start background task
    background_tasks.add_task(run_validation_task, validation_id, request)

    return ValidationResponse(
        validation_id=validation_id,
        status="started",
        message=f"Validation started for {request.symbol} with template {request.template}",
    )


@app.get("/api/validate/{validation_id}")
async def get_validation_result(validation_id: str):
    """Get validation result by ID."""
    if validation_id not in validation_status:
        raise HTTPException(status_code=404, detail="Validation not found")

    status = validation_status[validation_id]

    if status == "running":
        return {
            "validation_id": validation_id,
            "status": "running",
            "message": "Validation in progress...",
        }

    if validation_id in validation_results:
        return validation_results[validation_id]

    raise HTTPException(status_code=404, detail="Result not found")


@app.get("/api/strategies")
async def list_strategies():
    """List all completed validations."""
    completed = [
        result for result in validation_results.values()
        if result["status"] == "completed"
    ]

    # Convert to StrategyInfo format
    strategies = []
    for result in completed:
        metrics = result["metrics"]
        strategies.append({
            "id": result["validation_id"],
            "name": result["strategy_name"],
            "type": "Template",
            "status": "approved" if result["passed"] else "rejected",
            "sharpe": metrics.get("sharpe_ratio", 0.0),
            "dsr": metrics.get("dsr", 0.0),
            "annual_return": metrics.get("annual_return", 0.0),
        })

    return strategies


@app.get("/api/templates")
async def list_templates():
    """List available strategy templates."""
    templates = []

    # Get all template methods from StrategyTemplates
    for attr in dir(StrategyTemplates):
        if not attr.startswith("_") and callable(getattr(StrategyTemplates, attr)):
            templates.append({
                "name": attr,
                "display_name": attr.replace("_", " ").title(),
            })

    return templates


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
