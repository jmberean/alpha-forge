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
from alphaforge.factory import StrategyFactory
from alphaforge.factory.orchestrator import FactoryConfig

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

# Storage for factory runs
factory_results = {}
factory_status = {}

# Pipeline statistics (accumulated from runs)
pipeline_stats = {
    "total_generated": 0,
    "total_validated": 0,
    "total_passed": 0,
    "total_deployed": 0,
}


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


class FactoryRequest(BaseModel):
    symbol: str = "SPY"
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    population_size: int = 30
    generations: int = 5
    target_strategies: int = 50
    validate_top_n: int = 5


class FactoryResponse(BaseModel):
    factory_id: str
    status: str
    message: str


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


# Helper function to run strategy factory
async def run_factory_task(factory_id: str, request: FactoryRequest):
    """Background task to run strategy factory."""
    global pipeline_stats
    logs = []

    try:
        factory_status[factory_id] = "running"
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Strategy Factory")

        # Load data
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading market data for {request.symbol}...")
        loader = MarketDataLoader()
        data = loader.load(request.symbol, start=request.start_date, end=request.end_date)
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(data.df)} trading days")

        # Setup backtest engine
        engine = BacktestEngine()

        def fitness(strategy):
            try:
                result = engine.run(strategy, data)
                return result.metrics.sharpe_ratio
            except Exception:
                return -999

        # Configure factory
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Configuring genetic evolution...")
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]   Population: {request.population_size}")
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]   Generations: {request.generations}")

        config = FactoryConfig(
            genetic_population=request.population_size,
            genetic_generations=request.generations,
            target_strategies=request.target_strategies,
            min_fitness=-999,
        )

        # Run factory
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Running genetic evolution...")
        factory = StrategyFactory(fitness_function=fitness, config=config)
        pool = factory.generate()
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Generated {len(pool.strategies)} candidates")

        # Get top strategies
        top_strategies = factory.get_top_strategies(n=request.validate_top_n)
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Top {len(top_strategies)} strategies selected for validation")

        # Validate top strategies
        pipeline = ValidationPipeline()
        validated_strategies = []
        passed_count = 0

        for i, (strategy, fitness_score) in enumerate(top_strategies, 1):
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Validating #{i}: {strategy.name} (Sharpe: {fitness_score:.3f})")

            try:
                result = pipeline.validate(
                    strategy=strategy,
                    data=data,
                    n_trials=len(pool.strategies),
                    run_cpcv=True,
                    run_spa=False,
                    run_stress=True,
                )

                backtest = engine.run(strategy, data)
                dsr_val = result.dsr_result.dsr_pvalue if result.dsr_result else 0.0
                pbo_val = result.pbo_result.pbo if result.pbo_result else None
                stress_val = result.stress_result.pass_rate if result.stress_result else None

                status = "passed" if result.passed else "failed"
                if result.passed:
                    passed_count += 1
                logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]   {'PASSED' if result.passed else 'FAILED'}")

                validated_strategies.append({
                    "id": str(uuid.uuid4()),
                    "name": strategy.name,
                    "type": "Genetic",
                    "status": status,
                    "sharpe": backtest.metrics.sharpe_ratio,
                    "dsr": dsr_val,
                    "pbo": pbo_val,
                    "stress_pass_rate": stress_val,
                    "annual_return": backtest.metrics.annualized_return,
                    "max_drawdown": backtest.metrics.max_drawdown,
                    "total_return": backtest.metrics.total_return,
                    "fitness": fitness_score,
                })

                # Also add to global validation results
                vid = validated_strategies[-1]["id"]
                validation_results[vid] = {
                    "validation_id": vid,
                    "status": "completed",
                    "strategy_name": strategy.name,
                    "passed": result.passed,
                    "metrics": {
                        "sharpe_ratio": backtest.metrics.sharpe_ratio,
                        "dsr": dsr_val,
                        "pbo": pbo_val,
                        "annual_return": backtest.metrics.annualized_return,
                        "max_drawdown": backtest.metrics.max_drawdown,
                        "sortino_ratio": backtest.metrics.sortino_ratio,
                        "total_return": backtest.metrics.total_return,
                        "volatility": backtest.metrics.volatility,
                        "stress_pass_rate": stress_val,
                    },
                    "timestamp": datetime.now().isoformat(),
                    "logs": [],
                }
                validation_status[vid] = "completed"

            except Exception as e:
                logs.append(f"[{datetime.now().strftime('%H:%M:%S')}]   Error: {str(e)}")

        # Update pipeline stats
        pipeline_stats["total_generated"] += len(pool.strategies)
        pipeline_stats["total_validated"] += len(validated_strategies)
        pipeline_stats["total_passed"] += passed_count

        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ━━━ FACTORY COMPLETE ━━━")
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Generated: {len(pool.strategies)}")
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Validated: {len(validated_strategies)}")
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Passed: {passed_count}")

        factory_results[factory_id] = {
            "factory_id": factory_id,
            "status": "completed",
            "strategies": validated_strategies,
            "stats": {
                "generated": len(pool.strategies),
                "validated": len(validated_strategies),
                "passed": passed_count,
            },
            "timestamp": datetime.now().isoformat(),
            "logs": logs,
        }
        factory_status[factory_id] = "completed"

    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}")
        factory_results[factory_id] = {
            "factory_id": factory_id,
            "status": "failed",
            "strategies": [],
            "stats": {},
            "timestamp": datetime.now().isoformat(),
            "logs": logs,
        }
        factory_status[factory_id] = "failed"


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


@app.post("/api/factory", response_model=FactoryResponse)
async def run_factory(request: FactoryRequest, background_tasks: BackgroundTasks):
    """
    Run the strategy factory to generate and validate strategies.

    Uses genetic evolution to generate candidates, then validates the top performers.
    """
    factory_id = str(uuid.uuid4())

    # Start background task
    background_tasks.add_task(run_factory_task, factory_id, request)

    return FactoryResponse(
        factory_id=factory_id,
        status="started",
        message=f"Factory started for {request.symbol} with population {request.population_size}",
    )


@app.get("/api/factory/{factory_id}")
async def get_factory_result(factory_id: str):
    """Get factory result by ID."""
    if factory_id not in factory_status:
        raise HTTPException(status_code=404, detail="Factory run not found")

    status = factory_status[factory_id]

    if status == "running":
        # Return partial logs if available
        if factory_id in factory_results:
            return {
                "factory_id": factory_id,
                "status": "running",
                "message": "Factory in progress...",
                "logs": factory_results[factory_id].get("logs", []),
            }
        return {
            "factory_id": factory_id,
            "status": "running",
            "message": "Factory in progress...",
        }

    if factory_id in factory_results:
        return factory_results[factory_id]

    raise HTTPException(status_code=404, detail="Result not found")


@app.get("/api/pipeline-stats")
async def get_pipeline_stats():
    """Get accumulated pipeline statistics."""
    return {
        "stages": [
            {"name": "Generated", "count": pipeline_stats["total_generated"], "rate": 100},
            {"name": "Screened (DSR)", "count": pipeline_stats["total_validated"], "rate": (pipeline_stats["total_validated"] / max(1, pipeline_stats["total_generated"])) * 100},
            {"name": "Validated (CPCV)", "count": pipeline_stats["total_validated"], "rate": (pipeline_stats["total_validated"] / max(1, pipeline_stats["total_generated"])) * 100},
            {"name": "Stress Tested", "count": pipeline_stats["total_validated"], "rate": (pipeline_stats["total_validated"] / max(1, pipeline_stats["total_generated"])) * 100},
            {"name": "Passed", "count": pipeline_stats["total_passed"], "rate": (pipeline_stats["total_passed"] / max(1, pipeline_stats["total_validated"])) * 100},
            {"name": "Deployed", "count": pipeline_stats["total_deployed"], "rate": (pipeline_stats["total_deployed"] / max(1, pipeline_stats["total_passed"])) * 100},
        ],
        "totals": pipeline_stats,
    }


@app.get("/api/metrics/latest")
async def get_latest_metrics():
    """Get metrics from the most recent validation."""
    if not validation_results:
        return {
            "has_data": False,
            "metrics": [],
        }

    # Get most recent result
    latest = max(validation_results.values(), key=lambda x: x.get("timestamp", ""))
    metrics = latest.get("metrics", {})

    return {
        "has_data": True,
        "strategy_name": latest.get("strategy_name", "Unknown"),
        "passed": latest.get("passed", False),
        "timestamp": latest.get("timestamp", ""),
        "metrics": [
            {"name": "Deflated Sharpe Ratio", "value": metrics.get("dsr", 0), "threshold": "> 0.95", "unit": ""},
            {"name": "Probability of Backtest Overfitting", "value": metrics.get("pbo", 0), "threshold": "< 0.05", "unit": ""},
            {"name": "Sharpe Ratio", "value": metrics.get("sharpe_ratio", 0), "threshold": "> 1.0", "unit": ""},
            {"name": "Annual Return", "value": metrics.get("annual_return", 0) * 100, "threshold": "", "unit": "%"},
            {"name": "Max Drawdown", "value": abs(metrics.get("max_drawdown", 0)) * 100, "threshold": "", "unit": "%"},
            {"name": "Sortino Ratio", "value": metrics.get("sortino_ratio", 0), "threshold": "", "unit": ""},
            {"name": "Volatility", "value": metrics.get("volatility", 0) * 100, "threshold": "", "unit": "%"},
            {"name": "Total Return", "value": metrics.get("total_return", 0) * 100, "threshold": "", "unit": "%"},
        ],
    }


@app.get("/api/system/status")
async def get_system_status():
    """Get system status and health information."""
    total_validations = len(validation_results)
    passed = sum(1 for r in validation_results.values() if r.get("passed", False))

    return {
        "status": "online",
        "version": "MVP7",
        "total_validations": total_validations,
        "passed_validations": passed,
        "pass_rate": (passed / max(1, total_validations)) * 100,
        "factory_runs": len(factory_results),
        "pipeline_stats": pipeline_stats,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
