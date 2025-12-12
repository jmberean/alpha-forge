"""
Command-line interface for AlphaForge.

Provides commands for:
- Loading and caching market data
- Running backtests
- Validating strategies
- Managing experiments
"""

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import click

from alphaforge import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("alphaforge")


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(verbose: bool) -> None:
    """AlphaForge - Trading Strategy Validation Platform."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.argument("symbol")
@click.option(
    "--start",
    "-s",
    default=None,
    help="Start date (YYYY-MM-DD). Default: 5 years ago",
)
@click.option(
    "--end",
    "-e",
    default=None,
    help="End date (YYYY-MM-DD). Default: today",
)
@click.option("--no-cache", is_flag=True, help="Bypass cache and fetch fresh data")
def data(symbol: str, start: str, end: str, no_cache: bool) -> None:
    """Load and cache market data for a symbol."""
    from alphaforge.data.loader import MarketDataLoader

    # Default dates
    if start is None:
        start_date = date.today() - timedelta(days=5 * 365)
    else:
        start_date = date.fromisoformat(start)

    if end is None:
        end_date = date.today()
    else:
        end_date = date.fromisoformat(end)

    loader = MarketDataLoader()

    click.echo(f"Loading {symbol} from {start_date} to {end_date}...")

    try:
        data = loader.load(symbol, start_date, end_date, use_cache=not no_cache)
        click.echo(f"Loaded {len(data)} rows")
        click.echo(f"Date range: {data.df.index[0].date()} to {data.df.index[-1].date()}")
        click.echo(f"Columns: {list(data.df.columns)}")
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("symbol")
@click.option("--template", "-t", default="sma_crossover", help="Strategy template name")
@click.option("--start", "-s", default=None, help="Start date")
@click.option("--end", "-e", default=None, help="End date")
@click.option("--output", "-o", default=None, help="Output file for results")
def backtest(symbol: str, template: str, start: str, end: str, output: str) -> None:
    """Run a backtest on a symbol with a strategy template."""
    from alphaforge.data.loader import MarketDataLoader
    from alphaforge.strategy.templates import StrategyTemplates
    from alphaforge.backtest.engine import BacktestEngine

    # Load data
    loader = MarketDataLoader()

    if start is None:
        start_date = date.today() - timedelta(days=3 * 365)
    else:
        start_date = date.fromisoformat(start)

    if end is None:
        end_date = date.today()
    else:
        end_date = date.fromisoformat(end)

    click.echo(f"Loading {symbol}...")
    data = loader.load(symbol, start_date, end_date)

    # Get strategy
    try:
        strategy = StrategyTemplates.get_template(template)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Available templates:")
        for s in StrategyTemplates.all_templates():
            click.echo(f"  - {s.name.lower().replace(' ', '_')}")
        sys.exit(1)

    click.echo(f"Running backtest: {strategy.name}...")

    # Run backtest
    engine = BacktestEngine()
    result = engine.run(strategy, data)

    # Display results
    click.echo("\n" + "=" * 50)
    click.echo(result.summary())
    click.echo("=" * 50)

    # Save if output specified
    if output:
        output_path = Path(output)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        click.echo(f"\nResults saved to {output}")


@main.command()
@click.argument("symbol")
@click.option("--template", "-t", default="sma_crossover", help="Strategy template name")
@click.option("--start", "-s", default=None, help="Start date")
@click.option("--end", "-e", default=None, help="End date")
@click.option("--n-trials", "-n", default=1, help="Number of strategies tested (for DSR)")
@click.option("--run-cpcv/--no-cpcv", default=False, help="Run CPCV validation")
@click.option("--output", "-o", default=None, help="Output file for results")
def validate(
    symbol: str,
    template: str,
    start: str,
    end: str,
    n_trials: int,
    run_cpcv: bool,
    output: str,
) -> None:
    """Validate a strategy with full statistical tests."""
    from alphaforge.data.loader import MarketDataLoader
    from alphaforge.strategy.templates import StrategyTemplates
    from alphaforge.validation.pipeline import ValidationPipeline

    # Load data
    loader = MarketDataLoader()

    if start is None:
        start_date = date.today() - timedelta(days=5 * 365)
    else:
        start_date = date.fromisoformat(start)

    if end is None:
        end_date = date.today()
    else:
        end_date = date.fromisoformat(end)

    click.echo(f"Loading {symbol}...")
    data = loader.load(symbol, start_date, end_date)

    # Get strategy
    try:
        strategy = StrategyTemplates.get_template(template)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Validating: {strategy.name}")
    click.echo(f"  n_trials: {n_trials}")
    click.echo(f"  CPCV: {'enabled' if run_cpcv else 'disabled'}")

    # Run validation
    pipeline = ValidationPipeline()
    result = pipeline.validate(strategy, data, n_trials=n_trials, run_cpcv=run_cpcv)

    # Display results
    click.echo("\n" + result.summary())

    # Final verdict
    if result.auto_accept:
        click.secho("\nVERDICT: AUTO-ACCEPT", fg="green", bold=True)
    elif result.passed:
        click.secho("\nVERDICT: PASSED (requires manual review)", fg="yellow", bold=True)
    else:
        click.secho("\nVERDICT: FAILED", fg="red", bold=True)

    # Save if output specified
    if output:
        output_path = Path(output)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        click.echo(f"\nResults saved to {output}")


@main.command()
def templates() -> None:
    """List available strategy templates."""
    from alphaforge.strategy.templates import StrategyTemplates

    click.echo("Available Strategy Templates:\n")

    for strategy in StrategyTemplates.all_templates():
        click.echo(f"  {strategy.name}")
        click.echo(f"    ID: {strategy.name.lower().replace(' ', '_')}")
        click.echo(f"    Description: {strategy.description}")
        click.echo(f"    Tags: {', '.join(strategy.tags)}")
        click.echo("")


@main.command()
@click.option("--metric", "-m", default="sharpe_ratio", help="Metric to sort by")
@click.option("--limit", "-n", default=10, help="Number of results")
def leaderboard(metric: str, limit: int) -> None:
    """Show top strategies from experiment tracking."""
    try:
        from alphaforge.tracking.mlflow import ExperimentTracker

        tracker = ExperimentTracker()
        strategies = tracker.get_best_strategies(metric=metric, n=limit)

        if not strategies:
            click.echo("No strategies tracked yet.")
            return

        click.echo(f"\nTop {limit} Strategies by {metric}:\n")
        click.echo(f"{'Rank':<6}{'Name':<25}{metric:<15}{'Passed':<10}")
        click.echo("-" * 56)

        for i, s in enumerate(strategies, 1):
            passed = "Yes" if s["passed"] else "No"
            value = s[metric]
            value_str = f"{value:.4f}" if value else "N/A"
            click.echo(f"{i:<6}{s['strategy_name']:<25}{value_str:<15}{passed:<10}")

    except Exception as e:
        click.echo(f"Error accessing experiment tracker: {e}", err=True)
        click.echo("Make sure MLflow is configured correctly.")


@main.command()
def info() -> None:
    """Show AlphaForge installation info."""
    import numpy as np
    import pandas as pd

    click.echo(f"AlphaForge v{__version__}\n")
    click.echo("Dependencies:")
    click.echo(f"  numpy: {np.__version__}")
    click.echo(f"  pandas: {pd.__version__}")

    try:
        import scipy

        click.echo(f"  scipy: {scipy.__version__}")
    except ImportError:
        click.echo("  scipy: not installed")

    try:
        import mlflow

        click.echo(f"  mlflow: {mlflow.__version__}")
    except ImportError:
        click.echo("  mlflow: not installed")

    click.echo("\nData cache: data/cache/")
    click.echo("Feature cache: data/features/")


if __name__ == "__main__":
    main()
