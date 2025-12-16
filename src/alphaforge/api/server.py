"""
FastAPI server for AlphaForge.

Provides REST API endpoints to run validations and retrieve results.
"""

import os
import uuid
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from alphaforge.api.storage import Storage
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.data.loader import MarketDataLoader
from alphaforge.discovery import DiscoveryConfig, DiscoveryOrchestrator
from alphaforge.factory import StrategyFactory
from alphaforge.factory.orchestrator import FactoryConfig
from alphaforge.strategy.templates import StrategyTemplates
from alphaforge.validation.pipeline import ValidationPipeline

# Create FastAPI app
app = FastAPI(
    title="AlphaForge API",
    description="Production-grade systematic trading strategy validation",
    version="0.1.0",
)

# Enable CORS for frontend (configurable via environment variable)
CORS_ORIGINS = os.environ.get("ALPHAFORGE_CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistence
storage = Storage()

# In-memory storage for running status (ephemeral)
validation_status = {}
factory_status = {}
discovery_status = {}
discovery_logs = {}  # Separate storage for live logs
factory_logs = {}

# Pipeline statistics (loaded from and persisted to database)
pipeline_stats = storage.load_pipeline_stats()


# Request/Response Models
class ValidationRequest(BaseModel):
    symbol: str
    template: str
    start_date: str | None = "2020-01-01"
    end_date: str | None = "2023-12-31"
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
    equity_curve: list[dict] = []


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


class DiscoveryRequest(BaseModel):
    symbol: str = "SPY"
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    population_size: int = 100
    n_generations: int = 20
    n_objectives: int = 4
    min_sharpe: float = 0.5
    max_turnover: float = 0.2
    max_complexity: float = 0.7
    validation_split: float = 0.3


class DiscoveryResponse(BaseModel):
    discovery_id: str
    status: str
    message: str


# Helper function to run validation
async def run_validation_task(validation_id: str, request: ValidationRequest):
    """Background task to run validation."""
    logs = []

    try:
        # Update status
        validation_status[validation_id] = "running"
        logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] Starting validation for {request.symbol}"
        )

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
        logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Sharpe: {backtest_result.metrics.sharpe_ratio:.2f}"
        )

        # Always run benchmark for comparison chart
        logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] Calculating benchmark ({request.symbol})..."
        )
        benchmark_strategy = StrategyTemplates.buy_and_hold()
        benchmark_result = engine.run(benchmark_strategy, data)
        benchmark_returns = benchmark_result.returns

        # Prepare equity curve data for chart
        equity_curve_data = []
        strategy_equity = backtest_result.equity_curve
        benchmark_equity = benchmark_result.equity_curve

        # Align indexes and create records
        common_index = strategy_equity.index.intersection(benchmark_equity.index)

        # Downsample for frontend performance (max 500 points)
        if len(common_index) > 500:
            step = len(common_index) // 500
            common_index = common_index[::step]

        for date in common_index:
            equity_curve_data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "equity": float(strategy_equity.loc[date]),
                    "benchmark": float(benchmark_equity.loc[date]),
                }
            )

        # Run validation
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Running statistical validation...")
        pipeline = ValidationPipeline()

        validation_result = pipeline.validate(
            strategy=strategy,
            data=data,
            n_trials=request.n_trials,
            run_cpcv=request.run_cpcv,
            run_spa=request.run_spa,
            run_stress=request.run_stress,
            benchmark_returns=benchmark_returns,
            benchmark_name=request.symbol,
        )

        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ━━━ VALIDATION COMPLETE ━━━")
        logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] {'✓ PASSED' if validation_result.passed else '✗ FAILED'}"
        )

        # Save result to DB
        result_data = {
            "validation_id": validation_id,
            "status": "completed",
            "strategy_name": strategy.name,
            "passed": validation_result.passed,
            "metrics": {
                "sharpe_ratio": backtest_result.metrics.sharpe_ratio,
                "dsr": validation_result.dsr_result.dsr_pvalue
                if validation_result.dsr_result
                else 0.0,
                "prob_loss": validation_result.pbo_result.pbo
                if validation_result.pbo_result
                else None,
                "annual_return": backtest_result.metrics.annualized_return,
                "max_drawdown": backtest_result.metrics.max_drawdown,
                "sortino_ratio": backtest_result.metrics.sortino_ratio,
                "total_return": backtest_result.metrics.total_return,
                "volatility": backtest_result.metrics.volatility,
                "num_trades": backtest_result.metrics.num_trades,
                "win_rate": backtest_result.metrics.win_rate,
                "profit_factor": backtest_result.metrics.profit_factor,
                "avg_win": backtest_result.metrics.avg_win,
                "avg_loss": backtest_result.metrics.avg_loss,
            },
            "timestamp": datetime.now().isoformat(),
            "logs": logs,
            "equity_curve": equity_curve_data,
        }
        storage.save_validation(validation_id, result_data)
        validation_status[validation_id] = "completed"

    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Error: {str(e)}")
        error_data = {
            "validation_id": validation_id,
            "status": "failed",
            "strategy_name": request.template,
            "passed": False,
            "metrics": {},
            "timestamp": datetime.now().isoformat(),
            "logs": logs,
        }
        storage.save_validation(validation_id, error_data)
        validation_status[validation_id] = "failed"


# Helper function to run strategy factory
async def run_factory_task(factory_id: str, request: FactoryRequest):
    """Background task to run strategy factory."""
    global pipeline_stats
    logs = []

    try:
        factory_status[factory_id] = "running"
        factory_logs[factory_id] = []

        def log(msg: str) -> None:
            logs.append(msg)
            factory_logs[factory_id].append(msg)

        log(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Strategy Factory")

        # Load data
        log(f"[{datetime.now().strftime('%H:%M:%S')}] Loading market data for {request.symbol}...")
        loader = MarketDataLoader()
        data = loader.load(request.symbol, start=request.start_date, end=request.end_date)
        log(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(data.df)} trading days")

        # Setup backtest engine
        engine = BacktestEngine()

        def fitness(strategy):
            try:
                result = engine.run(strategy, data)
                return result.metrics.sharpe_ratio
            except Exception:
                return -999

        # Configure factory
        log(f"[{datetime.now().strftime('%H:%M:%S')}] Configuring genetic evolution...")
        log(f"[{datetime.now().strftime('%H:%M:%S')}]   Population: {request.population_size}")
        log(f"[{datetime.now().strftime('%H:%M:%S')}]   Generations: {request.generations}")

        config = FactoryConfig(
            genetic_population=request.population_size,
            genetic_generations=request.generations,
            target_strategies=request.target_strategies,
            min_fitness=-999,
        )

        # Run factory
        log(f"[{datetime.now().strftime('%H:%M:%S')}] Running genetic evolution...")
        factory = StrategyFactory(fitness_function=fitness, config=config)
        pool = factory.generate()
        log(f"[{datetime.now().strftime('%H:%M:%S')}] Generated {len(pool.strategies)} candidates")

        # Get top strategies
        top_strategies = factory.get_top_strategies(n=request.validate_top_n)
        log(
            f"[{datetime.now().strftime('%H:%M:%S')}] Top {len(top_strategies)} strategies selected for validation"
        )

        # Validate top strategies
        pipeline = ValidationPipeline()
        validated_strategies = []
        passed_count = 0

        for i, (strategy, fitness_score) in enumerate(top_strategies, 1):
            log(
                f"[{datetime.now().strftime('%H:%M:%S')}] Validating #{i}: {strategy.name} (Sharpe: {fitness_score:.3f})"
            )

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
                prob_loss_val = result.pbo_result.pbo if result.pbo_result else None
                stress_val = result.stress_result.pass_rate if result.stress_result else None

                status = "passed" if result.passed else "failed"
                if result.passed:
                    passed_count += 1
                log(
                    f"[{datetime.now().strftime('%H:%M:%S')}]   {'PASSED' if result.passed else 'FAILED'}"
                )

                validated_strategies.append(
                    {
                        "id": str(uuid.uuid4()),
                        "name": strategy.name,
                        "type": "Genetic",
                        "status": status,
                        "sharpe": backtest.metrics.sharpe_ratio,
                        "dsr": dsr_val,
                        "prob_loss": prob_loss_val,
                        "stress_pass_rate": stress_val,
                        "annual_return": backtest.metrics.annualized_return,
                        "max_drawdown": backtest.metrics.max_drawdown,
                        "total_return": backtest.metrics.total_return,
                        "fitness": fitness_score,
                    }
                )

                # Persist these individual validation results too!
                vid = validated_strategies[-1]["id"]
                val_data = {
                    "validation_id": vid,
                    "status": "completed",
                    "strategy_name": strategy.name,
                    "passed": result.passed,
                    "metrics": {
                        "sharpe_ratio": backtest.metrics.sharpe_ratio,
                        "dsr": dsr_val,
                        "prob_loss": prob_loss_val,
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
                storage.save_validation(vid, val_data)

            except Exception as e:
                log(f"[{datetime.now().strftime('%H:%M:%S')}]   Error: {str(e)}")

        # Update pipeline stats
        pipeline_stats["total_generated"] += len(pool.strategies)
        pipeline_stats["total_validated"] += len(validated_strategies)
        pipeline_stats["total_passed"] += passed_count
        storage.save_pipeline_stats(pipeline_stats)

        log(f"[{datetime.now().strftime('%H:%M:%S')}] ━━━ FACTORY COMPLETE ━━━")
        log(f"[{datetime.now().strftime('%H:%M:%S')}] Generated: {len(pool.strategies)}")
        log(f"[{datetime.now().strftime('%H:%M:%S')}] Validated: {len(validated_strategies)}")
        log(f"[{datetime.now().strftime('%H:%M:%S')}] Passed: {passed_count}")

        factory_result = {
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

        storage.save_factory_run(
            factory_id=factory_id,
            status="completed",
            config=request.model_dump(),
            result=factory_result,
        )

        factory_status[factory_id] = "completed"

    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}")
        storage.save_factory_run(
            factory_id=factory_id,
            status="failed",
            config=request.model_dump(),
            result={
                "factory_id": factory_id,
                "status": "failed",
                "strategies": [],
                "stats": {},
                "timestamp": datetime.now().isoformat(),
                "logs": logs,
                "error": str(e),
            },
        )
        factory_status[factory_id] = "failed"


# Helper function to add log and store immediately
def add_discovery_log(discovery_id: str, message: str):
    """Add a log message and store it immediately for live updates."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    if discovery_id not in discovery_logs:
        discovery_logs[discovery_id] = []
    discovery_logs[discovery_id].append(log_entry)
    return log_entry


# Helper function to run discovery
async def run_discovery_task(discovery_id: str, request: DiscoveryRequest):
    """Background task to run strategy discovery."""
    global pipeline_stats

    # Initialize logs storage
    discovery_logs[discovery_id] = []

    def log(msg: str):
        add_discovery_log(discovery_id, msg)

    try:
        discovery_status[discovery_id] = "running"
        log("Starting Strategy Discovery System")
        log("Multi-Objective Genetic Programming (NSGA-III)")

        # Load data
        log(f"Loading market data for {request.symbol}...")
        loader = MarketDataLoader()
        data = loader.load(request.symbol, start=request.start_date, end=request.end_date)
        log(f"✓ Loaded {len(data.df)} trading days")

        # Configure discovery
        log("Configuring discovery system...")
        log(f"  Population: {request.population_size}")
        log(f"  Generations: {request.n_generations}")
        log(f"  Objectives: {request.n_objectives} (Sharpe, MaxDD, Turnover, Complexity)")
        log(f"  Validation split: {request.validation_split:.0%}")

        config = DiscoveryConfig(
            population_size=request.population_size,
            n_generations=request.n_generations,
            n_objectives=request.n_objectives,
            min_sharpe=request.min_sharpe,
            max_turnover=request.max_turnover,
            max_complexity=request.max_complexity,
            validation_split=request.validation_split,
        )

        # Run discovery with progress callback
        log("Initializing NSGA-III optimizer...")

        def on_generation(gen: int, stats: dict):
            """Callback for each generation."""
            fitness = stats.get("fitness", {})
            best_sharpe = (
                fitness.get("max", {}).get("sharpe", 0)
                if isinstance(fitness.get("max"), dict)
                else 0
            )
            pareto_size = stats.get("pareto_front_size", 0)
            log(
                f"Gen {gen:3d}/{request.n_generations} | Pareto: {pareto_size} | Unique: {stats.get('unique_formulas', 0)}"
            )

        log("Running multi-objective evolution...")
        orchestrator = DiscoveryOrchestrator(data.df, config)
        result = orchestrator.discover(on_generation=on_generation)

        log("━━━ DISCOVERY COMPLETE ━━━")
        log(f"Generations: {result.n_generations}")
        log(f"Pareto front: {len(result.pareto_front)} strategies")
        log(f"Factor zoo: {len(result.factor_zoo)} validated formulas")

        # Run statistical validation on Pareto front strategies
        log("")
        log("Running statistical validation on Pareto front...")
        pipeline = ValidationPipeline()
        validated_count = 0
        passed_count = 0

        # Convert Pareto front to StrategyGenome for validation
        strategy_genomes = orchestrator.to_strategy_genomes(result.pareto_front)

        # Only validate top strategies to keep reasonable runtime (e.g., top 10)
        max_validate = min(10, len(strategy_genomes))
        log(
            f"Validating top {max_validate} Pareto strategies (n_trials={request.population_size * request.n_generations})..."
        )

        validated_strategies = []
        for i, strategy in enumerate(strategy_genomes[:max_validate]):
            try:
                val_result = pipeline.validate(
                    strategy=strategy,
                    data=data,
                    n_trials=request.population_size * request.n_generations,
                    run_cpcv=False,  # Skip CPCV for speed (too slow for 10 strategies)
                    run_spa=False,
                    run_stress=False,
                )

                validated_count += 1
                if val_result.passed:
                    passed_count += 1

                log(
                    f"  #{i + 1}: {'PASSED' if val_result.passed else 'FAILED'} - DSR: {val_result.dsr_result.dsr_pvalue:.4f}"
                )

                validated_strategies.append(
                    {
                        "index": i,
                        "passed": val_result.passed,
                        "dsr": val_result.dsr_result.dsr_pvalue if val_result.dsr_result else 0.0,
                    }
                )
            except Exception as e:
                log(f"  #{i + 1}: Error validating - {str(e)}")

        log(f"✓ Validation complete: {passed_count}/{validated_count} passed DSR screening")

        # Convert result to serializable format
        pareto_front = []
        for i, ind in enumerate(result.pareto_front):
            # Find validation result for this strategy if available
            validation_info = None
            for val in validated_strategies:
                if val["index"] == i:
                    validation_info = {
                        "passed": val["passed"],
                        "dsr": val["dsr"],
                    }
                    break

            pareto_front.append(
                {
                    "formula": ind.genome.tree.formula,
                    "size": ind.genome.tree.size,
                    "depth": ind.genome.tree.depth,
                    "complexity": ind.genome.tree.complexity_score(),
                    "fitness": {
                        "sharpe": ind.fitness.get("sharpe", 0.0),
                        "drawdown": ind.fitness.get("drawdown", 0.0),
                        "turnover": ind.fitness.get("turnover", 0.0),
                        "complexity": ind.fitness.get("complexity", 0.0),
                    },
                    "validation": validation_info,
                }
            )

        best_by_objective = {}
        for obj_name, ind in result.best_by_objective.items():
            best_by_objective[obj_name] = {
                "formula": ind.genome.tree.formula,
                "size": ind.genome.tree.size,
                "depth": ind.genome.tree.depth,
                "complexity": ind.genome.tree.complexity_score(),
                "fitness": {
                    "sharpe": ind.fitness.get("sharpe", 0.0),
                    "drawdown": ind.fitness.get("drawdown", 0.0),
                    "turnover": ind.fitness.get("turnover", 0.0),
                    "complexity": ind.fitness.get("complexity", 0.0),
                },
            }

        factor_zoo = [tree.formula for tree in result.factor_zoo]

        # Update pipeline stats
        pipeline_stats["total_generated"] += request.population_size * request.n_generations
        pipeline_stats["total_validated"] += validated_count
        pipeline_stats["total_passed"] += passed_count
        storage.save_pipeline_stats(pipeline_stats)

        discovery_result = {
            "discovery_id": discovery_id,
            "status": "completed",
            "pareto_front": pareto_front,
            "factor_zoo": factor_zoo,
            "best_by_objective": best_by_objective,
            "ensemble_weights": result.ensemble_weights,
            "generation_stats": result.generation_stats,
            "n_generations": result.n_generations,
            "true_pbo": {
                "pbo": result.true_pbo.pbo if result.true_pbo else None,
                "rank_correlation": result.true_pbo.rank_correlation if result.true_pbo else None,
                "passed": result.true_pbo.passed if result.true_pbo else None,
            }
            if result.true_pbo
            else None,
            "timestamp": datetime.now().isoformat(),
            "logs": discovery_logs.get(discovery_id, []),
        }
        storage.save_discovery_run(
            discovery_id=discovery_id,
            status="completed",
            config=request.model_dump(),
            result=discovery_result,
        )
        discovery_status[discovery_id] = "completed"

    except Exception as e:
        import traceback

        log(f"✗ Error: {str(e)}")
        log(traceback.format_exc())
        storage.save_discovery_run(
            discovery_id=discovery_id,
            status="failed",
            config=request.model_dump(),
            result={
                "discovery_id": discovery_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "logs": discovery_logs.get(discovery_id, []),
            },
        )
        discovery_status[discovery_id] = "failed"


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
    # Check running status (in-memory) first
    if validation_id in validation_status:
        status = validation_status[validation_id]
        if status == "running":
            return {
                "validation_id": validation_id,
                "status": "running",
                "message": "Validation in progress...",
            }

    # Check persistence
    result = storage.get_validation(validation_id)
    if result:
        return result

    raise HTTPException(status_code=404, detail="Result not found")


@app.get("/api/strategies")
async def list_strategies():
    """List all completed validations."""
    completed = storage.list_validations()

    # Convert to StrategyInfo format
    strategies = []
    for result in completed:
        metrics = result.get("metrics", {})
        strategies.append(
            {
                "id": result["validation_id"],
                "name": result["strategy_name"],
                "type": "Template",
                "status": "approved" if result["passed"] else "rejected",
                "sharpe": metrics.get("sharpe_ratio", 0.0),
                "dsr": metrics.get("dsr", 0.0),
                "annual_return": metrics.get("annual_return", 0.0),
            }
        )

    return strategies


@app.get("/api/templates")
async def list_templates():
    """List available strategy templates."""
    templates = []

    # Get all template methods from StrategyTemplates
    for attr in dir(StrategyTemplates):
        if not attr.startswith("_") and callable(getattr(StrategyTemplates, attr)):
            templates.append(
                {
                    "name": attr,
                    "display_name": attr.replace("_", " ").title(),
                }
            )

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
        return {
            "factory_id": factory_id,
            "status": "running",
            "message": "Factory in progress...",
            "logs": factory_logs.get(factory_id, []),
        }

    result = storage.get_factory_run(factory_id)
    if result:
        return result

    raise HTTPException(status_code=404, detail="Result not found")


@app.get("/api/pipeline-stats")
async def get_pipeline_stats():
    """Get accumulated pipeline statistics."""
    return {
        "stages": [
            {"name": "Generated", "count": pipeline_stats["total_generated"], "rate": 100},
            {
                "name": "Screened (DSR)",
                "count": pipeline_stats["total_validated"],
                "rate": (
                    pipeline_stats["total_validated"] / max(1, pipeline_stats["total_generated"])
                )
                * 100,
            },
            {
                "name": "Validated (CPCV)",
                "count": pipeline_stats["total_validated"],
                "rate": (
                    pipeline_stats["total_validated"] / max(1, pipeline_stats["total_generated"])
                )
                * 100,
            },
            {
                "name": "Stress Tested",
                "count": pipeline_stats["total_validated"],
                "rate": (
                    pipeline_stats["total_validated"] / max(1, pipeline_stats["total_generated"])
                )
                * 100,
            },
            {
                "name": "Passed",
                "count": pipeline_stats["total_passed"],
                "rate": (pipeline_stats["total_passed"] / max(1, pipeline_stats["total_validated"]))
                * 100,
            },
            {
                "name": "Deployed",
                "count": pipeline_stats["total_deployed"],
                "rate": (pipeline_stats["total_deployed"] / max(1, pipeline_stats["total_passed"]))
                * 100,
            },
        ],
        "totals": pipeline_stats,
    }


@app.get("/api/metrics/latest")
async def get_latest_metrics():
    """Get metrics from the most recent validation."""
    # Fetch latest from DB
    validations = storage.list_validations()
    if not validations:
        return {
            "has_data": False,
            "metrics": [],
        }

    # Get most recent result
    latest = validations[0]  # list_validations sorts by desc timestamp
    metrics = latest.get("metrics", {})

    return {
        "has_data": True,
        "strategy_name": latest.get("strategy_name", "Unknown"),
        "passed": latest.get("passed", False),
        "timestamp": latest.get("timestamp", ""),
        "metrics": [
            {
                "name": "Deflated Sharpe Ratio",
                "value": metrics.get("dsr", 0),
                "threshold": "> 0.95",
                "unit": "",
            },
            {
                "name": "Probability of Loss",
                "value": metrics.get("prob_loss", 0),
                "threshold": "< 0.05",
                "unit": "",
            },
            {
                "name": "Sharpe Ratio",
                "value": metrics.get("sharpe_ratio", 0),
                "threshold": "> 1.0",
                "unit": "",
            },
            {
                "name": "Annual Return",
                "value": metrics.get("annual_return", 0) * 100,
                "threshold": "",
                "unit": "%",
            },
            {
                "name": "Max Drawdown",
                "value": abs(metrics.get("max_drawdown", 0)) * 100,
                "threshold": "",
                "unit": "%",
            },
            {
                "name": "Sortino Ratio",
                "value": metrics.get("sortino_ratio", 0),
                "threshold": "",
                "unit": "",
            },
            {
                "name": "Volatility",
                "value": metrics.get("volatility", 0) * 100,
                "threshold": "",
                "unit": "%",
            },
            {
                "name": "Total Return",
                "value": metrics.get("total_return", 0) * 100,
                "threshold": "",
                "unit": "%",
            },
        ],
    }


@app.get("/api/system/status")
async def get_system_status():
    """Get system status and health information."""
    # Get stats from DB
    validations = storage.list_validations()
    total_validations = len(validations)
    passed = sum(1 for r in validations if r.get("passed", False))
    factory_runs = storage.list_factory_runs()
    discovery_runs = storage.list_discovery_runs()

    return {
        "status": "online",
        "version": "MVP8",
        "total_validations": total_validations,
        "passed_validations": passed,
        "pass_rate": (passed / max(1, total_validations)) * 100,
        "factory_runs": len(factory_runs),
        "discovery_runs": len(discovery_runs),
        "pipeline_stats": pipeline_stats,
    }


@app.post("/api/discovery", response_model=DiscoveryResponse)
async def run_discovery(request: DiscoveryRequest, background_tasks: BackgroundTasks):
    """
    Run the strategy discovery system.

    Uses multi-objective genetic programming (NSGA-III) to evolve expression trees.
    Returns Pareto front of non-dominated strategies.
    """
    discovery_id = str(uuid.uuid4())

    # Start background task
    background_tasks.add_task(run_discovery_task, discovery_id, request)

    return DiscoveryResponse(
        discovery_id=discovery_id,
        status="started",
        message=f"Discovery started for {request.symbol} with population {request.population_size}",
    )


@app.get("/api/discovery/{discovery_id}")
async def get_discovery_result(discovery_id: str):
    """Get discovery result by ID."""
    if discovery_id not in discovery_status:
        raise HTTPException(status_code=404, detail="Discovery run not found")

    status = discovery_status[discovery_id]

    if status == "running":
        # Return live logs from discovery_logs
        return {
            "discovery_id": discovery_id,
            "status": "running",
            "message": "Discovery in progress...",
            "logs": discovery_logs.get(discovery_id, []),
        }

    result = storage.get_discovery_run(discovery_id)
    if result:
        return result

    raise HTTPException(status_code=404, detail="Result not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
