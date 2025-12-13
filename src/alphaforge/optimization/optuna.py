"""
Optuna-based Bayesian optimizer.

Uses Optuna's Tree-structured Parzen Estimator (TPE) for efficient
parameter optimization.
"""

from typing import Any

import optuna

from alphaforge.optimization.base import Optimizer, OptimizationResult


class OptunaOptimizer(Optimizer):
    """
    Optuna Bayesian optimizer.

    Uses Tree-structured Parzen Estimator (TPE) to efficiently search
    parameter space by building a probabilistic model.
    """

    def __init__(
        self,
        *args,
        sampler: str = "TPE",
        seed: int | None = None,
        show_progress: bool = False,
        **kwargs,
    ):
        """
        Initialize Optuna optimizer.

        Args:
            *args: Passed to Optimizer
            sampler: Optuna sampler type ('TPE', 'Random', 'Grid')
            seed: Random seed for reproducibility
            show_progress: Show progress bar during optimization
            **kwargs: Passed to Optimizer
        """
        super().__init__(*args, **kwargs)
        self.sampler_name = sampler
        self.seed = seed
        self.show_progress = show_progress

        # Create sampler
        if sampler == "TPE":
            self.sampler = optuna.samplers.TPESampler(seed=seed)
        elif sampler == "Random":
            self.sampler = optuna.samplers.RandomSampler(seed=seed)
        elif sampler == "Grid":
            # For grid search, we need to define search space
            # This is handled when creating the study
            self.sampler = None
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

    def optimize(self, n_trials: int, **kwargs) -> OptimizationResult:
        """
        Run Optuna optimization.

        Args:
            n_trials: Number of optimization trials
            **kwargs: Additional Optuna study arguments

        Returns:
            OptimizationResult with best parameters
        """
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create study
        direction = "maximize" if self.maximize else "minimize"

        if self.sampler is not None:
            study = optuna.create_study(
                direction=direction,
                sampler=self.sampler,
            )
        else:
            study = optuna.create_study(direction=direction)

        # Define objective function for Optuna
        def optuna_objective(trial: optuna.Trial) -> float:
            # Suggest parameters based on parameter space
            params = {}
            for p in self.parameter_space:
                if p.type == "int":
                    if p.log_scale:
                        params[p.name] = trial.suggest_int(
                            p.name, int(p.low), int(p.high), log=True
                        )
                    else:
                        params[p.name] = trial.suggest_int(
                            p.name, int(p.low), int(p.high)
                        )
                elif p.type == "float":
                    if p.log_scale:
                        params[p.name] = trial.suggest_float(
                            p.name, p.low, p.high, log=True
                        )
                    else:
                        params[p.name] = trial.suggest_float(p.name, p.low, p.high)
                elif p.type == "categorical":
                    params[p.name] = trial.suggest_categorical(p.name, p.choices)

            # Evaluate objective (Optuna handles maximize/minimize)
            score = self.objective(params)
            return score

        # Run optimization
        study.optimize(
            optuna_objective,
            n_trials=n_trials,
            show_progress_bar=self.show_progress,
            **kwargs,
        )

        # Extract results
        best_params = study.best_params
        best_score = study.best_value

        # Extract all trials
        trials = []
        scores = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials.append({
                    "params": trial.params.copy(),
                    "score": trial.value,
                })
                scores.append(trial.value)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            trials=trials,
            scores=scores,
            n_trials=len(study.trials),
            optimizer_name=f"Optuna-{self.sampler_name}",
        )
