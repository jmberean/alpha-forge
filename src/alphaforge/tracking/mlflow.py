"""
MLflow experiment tracking integration.

Tracks all strategy evaluations for reproducibility and audit trails.
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import json
import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

from alphaforge.strategy.genome import StrategyGenome
from alphaforge.backtest.engine import BacktestResult
from alphaforge.validation.pipeline import ValidationResult

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    MLflow-based experiment tracking for AlphaForge.

    Tracks:
    - Strategy configurations
    - Backtest parameters
    - Performance metrics
    - Validation results
    """

    DEFAULT_EXPERIMENT = "alphaforge-strategies"
    DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT,
        tracking_uri: Optional[str] = None,
    ) -> None:
        """
        Initialize experiment tracker.

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI (defaults to local SQLite)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", self.DEFAULT_TRACKING_URI
        )

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        self.experiment_id = self._get_or_create_experiment()

        self.client = MlflowClient()

    def _get_or_create_experiment(self) -> str:
        """Get existing experiment or create new one."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment:
            return experiment.experiment_id

        return mlflow.create_experiment(
            self.experiment_name,
            tags={"project": "alphaforge", "version": "1.0"},
        )

    def log_backtest(
        self,
        strategy: StrategyGenome,
        result: BacktestResult,
        data_symbol: str,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Log a backtest run.

        Args:
            strategy: Strategy that was backtested
            result: Backtest results
            data_symbol: Symbol used for backtest
            tags: Additional tags

        Returns:
            MLflow run ID
        """
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"{strategy.name}_{strategy.id}",
        ) as run:
            # Log strategy parameters
            mlflow.log_params(
                {
                    "strategy_id": strategy.id,
                    "strategy_name": strategy.name,
                    "strategy_version": strategy.version,
                    "data_symbol": data_symbol,
                    "sizing_method": strategy.sizing_method.value,
                    "max_position_pct": strategy.max_position_pct,
                    "stop_loss_pct": strategy.stop_loss_pct or "none",
                }
            )

            # Log strategy-specific parameters
            for key, value in strategy.parameters.items():
                mlflow.log_param(f"param_{key}", value)

            # Log metrics
            metrics = result.metrics
            mlflow.log_metrics(
                {
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "annualized_return": metrics.annualized_return,
                    "volatility": metrics.volatility,
                    "max_drawdown": metrics.max_drawdown,
                    "sortino_ratio": metrics.sortino_ratio,
                    "num_trades": metrics.num_trades,
                    "win_rate": metrics.win_rate,
                    "profit_factor": metrics.profit_factor,
                    "trading_days": metrics.trading_days,
                }
            )

            # Log tags
            mlflow.set_tags(
                {
                    "strategy_type": ",".join(strategy.tags) if strategy.tags else "untagged",
                    "backtest_start": result.start_date.isoformat(),
                    "backtest_end": result.end_date.isoformat(),
                    **(tags or {}),
                }
            )

            # Log strategy genome as artifact
            genome_path = Path("/tmp/strategy_genome.json")
            genome_path.write_text(strategy.to_json())
            mlflow.log_artifact(str(genome_path), "strategy")

            return run.info.run_id

    def log_validation(
        self,
        validation_result: ValidationResult,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Log a full validation run.

        Args:
            validation_result: Validation results
            tags: Additional tags

        Returns:
            MLflow run ID
        """
        strategy = validation_result.strategy
        backtest = validation_result.backtest_result
        dsr = validation_result.dsr_result

        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"validation_{strategy.name}_{strategy.id}",
        ) as run:
            # Log parameters
            mlflow.log_params(
                {
                    "strategy_id": strategy.id,
                    "strategy_name": strategy.name,
                    "n_trials": validation_result.n_trials,
                    "cpcv_n_splits": validation_result.thresholds.cpcv_n_splits,
                    "cpcv_test_splits": validation_result.thresholds.cpcv_test_splits,
                }
            )

            # Log backtest metrics
            metrics = backtest.metrics
            mlflow.log_metrics(
                {
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "annualized_return": metrics.annualized_return,
                    "volatility": metrics.volatility,
                    "max_drawdown": metrics.max_drawdown,
                }
            )

            # Log DSR metrics
            mlflow.log_metrics(
                {
                    "dsr_pvalue": dsr.dsr_pvalue,
                    "expected_max_sharpe": dsr.expected_max_sharpe,
                    "sharpe_std_error": dsr.sharpe_std_error,
                }
            )

            # Log CPCV metrics if available
            if validation_result.cpcv_result:
                cpcv = validation_result.cpcv_result
                mlflow.log_metrics(
                    {
                        "cpcv_mean_sharpe": cpcv.mean_sharpe,
                        "cpcv_std_sharpe": cpcv.std_sharpe,
                        "cpcv_pbo": cpcv.pbo,
                        "cpcv_n_combinations": cpcv.n_combinations,
                    }
                )

            # Log PBO metrics if available
            if validation_result.pbo_result:
                pbo = validation_result.pbo_result
                mlflow.log_metrics(
                    {
                        "pbo": pbo.pbo,
                        "pbo_logit": pbo.logit_pbo if pbo.logit_pbo != float("inf") else 10,
                    }
                )

            # Log pass/fail status
            mlflow.set_tags(
                {
                    "passed": str(validation_result.passed),
                    "auto_accept": str(validation_result.auto_accept),
                    **(tags or {}),
                }
            )

            # Log validation report as artifact
            report_path = Path("/tmp/validation_report.txt")
            report_path.write_text(validation_result.summary())
            mlflow.log_artifact(str(report_path), "reports")

            # Log full result as JSON
            result_path = Path("/tmp/validation_result.json")
            result_path.write_text(json.dumps(validation_result.to_dict(), indent=2))
            mlflow.log_artifact(str(result_path), "reports")

            return run.info.run_id

    def get_best_strategies(
        self,
        metric: str = "sharpe_ratio",
        n: int = 10,
        filter_passed: bool = True,
    ) -> list[dict]:
        """
        Get top N strategies by a metric.

        Args:
            metric: Metric to sort by
            n: Number of strategies to return
            filter_passed: Only include passed strategies

        Returns:
            List of strategy info dicts
        """
        # Search runs
        filter_string = ""
        if filter_passed:
            filter_string = "tags.passed = 'True'"

        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=[f"metrics.{metric} DESC"],
            max_results=n,
        )

        results = []
        for run in runs:
            results.append(
                {
                    "run_id": run.info.run_id,
                    "strategy_name": run.data.params.get("strategy_name"),
                    "strategy_id": run.data.params.get("strategy_id"),
                    metric: run.data.metrics.get(metric),
                    "passed": run.data.tags.get("passed") == "True",
                }
            )

        return results

    def get_run(self, run_id: str) -> dict:
        """Get detailed info for a specific run."""
        run = self.client.get_run(run_id)
        return {
            "run_id": run_id,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
            "start_time": run.info.start_time,
            "status": run.info.status,
        }

    def compare_strategies(self, run_ids: list[str]) -> dict:
        """Compare multiple strategy runs."""
        comparison = {}

        for run_id in run_ids:
            run = self.client.get_run(run_id)
            name = run.data.params.get("strategy_name", run_id)
            comparison[name] = {
                "sharpe_ratio": run.data.metrics.get("sharpe_ratio"),
                "dsr_pvalue": run.data.metrics.get("dsr_pvalue"),
                "pbo": run.data.metrics.get("pbo"),
                "annualized_return": run.data.metrics.get("annualized_return"),
                "max_drawdown": run.data.metrics.get("max_drawdown"),
            }

        return comparison
