"""
Deflated Sharpe Ratio (DSR) implementation.

From Bailey & Lopez de Prado (2014):
"The Deflated Sharpe Ratio"

DSR accounts for:
- Multiple testing (number of trials)
- Non-normality (skewness, kurtosis)
- Serial correlation
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class DSRResult:
    """Result of DSR calculation."""

    sharpe_ratio: float
    deflated_sharpe: float
    dsr_pvalue: float
    expected_max_sharpe: float
    sharpe_std_error: float
    n_trials: int
    sample_size: int
    skewness: float
    kurtosis: float
    passed: bool

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"DSRResult({status})\n"
            f"  Observed Sharpe: {self.sharpe_ratio:.3f}\n"
            f"  Expected Max (noise): {self.expected_max_sharpe:.3f}\n"
            f"  DSR p-value: {self.dsr_pvalue:.4f}\n"
            f"  n_trials: {self.n_trials}, T: {self.sample_size}"
        )


class DeflatedSharpeRatio:
    """
    Calculate the Deflated Sharpe Ratio.

    The DSR tells us the probability that the observed Sharpe ratio
    exceeds the expected maximum Sharpe ratio from random trials.

    Key insight: If you test 10,000 random strategies, the best one
    will have a Sharpe of ~4.3 purely by chance (expected max).
    DSR deflates the observed Sharpe to account for this.

    Formula:
        DSR = Φ((SR - E[SR_max]) / SE[SR])

    Where:
        - SR = observed Sharpe ratio
        - E[SR_max] = expected maximum Sharpe under null
        - SE[SR] = standard error of Sharpe estimate
        - Φ = standard normal CDF
    """

    # Euler-Mascheroni constant
    EULER_MASCHERONI = 0.5772156649

    def __init__(self, confidence_threshold: float = 0.95) -> None:
        """
        Initialize DSR calculator.

        Args:
            confidence_threshold: DSR threshold for passing (default 0.95)
        """
        self.confidence_threshold = confidence_threshold

    def calculate(
        self,
        sharpe_ratio: float,
        n_trials: int,
        sample_size: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> DSRResult:
        """
        Calculate the Deflated Sharpe Ratio.

        Args:
            sharpe_ratio: Observed annualized Sharpe ratio
            n_trials: Number of strategies tested (including this one)
            sample_size: Number of return observations (T)
            skewness: Skewness of returns (0 for normal)
            kurtosis: Kurtosis of returns (3 for normal)

        Returns:
            DSRResult with all calculations
        """
        # Expected maximum Sharpe from N random trials
        e_max_sharpe = self._expected_max_sharpe(n_trials)

        # Standard error of Sharpe ratio
        se_sharpe = self._sharpe_std_error(
            sharpe_ratio, sample_size, skewness, kurtosis
        )

        # Calculate DSR (probability observed SR > expected max from noise)
        if se_sharpe > 0:
            z_score = (sharpe_ratio - e_max_sharpe) / se_sharpe
            dsr = stats.norm.cdf(z_score)
        else:
            dsr = 0.0 if sharpe_ratio <= e_max_sharpe else 1.0

        return DSRResult(
            sharpe_ratio=sharpe_ratio,
            deflated_sharpe=dsr,
            dsr_pvalue=dsr,
            expected_max_sharpe=e_max_sharpe,
            sharpe_std_error=se_sharpe,
            n_trials=n_trials,
            sample_size=sample_size,
            skewness=skewness,
            kurtosis=kurtosis,
            passed=dsr >= self.confidence_threshold,
        )

    def calculate_from_returns(
        self,
        returns: np.ndarray,
        n_trials: int,
        annualization_factor: int = 252,
    ) -> DSRResult:
        """
        Calculate DSR from returns array.

        Args:
            returns: Array of returns
            n_trials: Number of strategies tested
            annualization_factor: Trading days per year

        Returns:
            DSRResult
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            raise ValueError("Need at least 2 returns")

        # Calculate statistics
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0:
            sharpe = 0.0
        else:
            sharpe = (mean_ret / std_ret) * np.sqrt(annualization_factor)

        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns) + 3)  # scipy returns excess kurtosis

        return self.calculate(
            sharpe_ratio=sharpe,
            n_trials=n_trials,
            sample_size=len(returns),
            skewness=skewness,
            kurtosis=kurtosis,
        )

    def _expected_max_sharpe(self, n_trials: int) -> float:
        """
        Calculate expected maximum Sharpe ratio from N random strategies.

        Uses the approximation from Lopez de Prado:
            E[SR_max] ≈ (1-γ) * Φ^(-1)(1 - 1/N) + γ * Φ^(-1)(1 - 1/(N*e))

        Where γ is the Euler-Mascheroni constant.

        For large N, this simplifies to: E[SR_max] ≈ √(2 * ln(N))
        """
        if n_trials <= 1:
            return 0.0

        gamma = self.EULER_MASCHERONI

        # Quantile calculations
        q1 = 1 - 1 / n_trials
        q2 = 1 - 1 / (n_trials * np.e)

        # Clamp to valid range for ppf
        q1 = min(max(q1, 1e-10), 1 - 1e-10)
        q2 = min(max(q2, 1e-10), 1 - 1e-10)

        z1 = stats.norm.ppf(q1)
        z2 = stats.norm.ppf(q2)

        return (1 - gamma) * z1 + gamma * z2

    def _sharpe_std_error(
        self,
        sharpe: float,
        sample_size: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Calculate standard error of Sharpe ratio.

        Accounts for non-normality using skewness and kurtosis.

        Formula:
            SE[SR] = sqrt((1 + 0.5*SR² - skew*SR + (kurt-3)/4 * SR²) / T)
        """
        if sample_size <= 1:
            return float("inf")

        # Adjust for excess kurtosis (scipy uses excess, we need raw)
        excess_kurtosis = kurtosis - 3

        variance_factor = (
            1
            + 0.5 * sharpe**2
            - skewness * sharpe
            + (excess_kurtosis / 4) * sharpe**2
        )

        # Ensure non-negative
        variance_factor = max(variance_factor, 0.01)

        return np.sqrt(variance_factor / sample_size)

    @staticmethod
    def quick_expected_max_sharpe(n_trials: int) -> float:
        """
        Quick approximation of expected max Sharpe.

        E[SR_max] ≈ √(2 * ln(N))

        Examples:
            N = 100 → E[SR_max] ≈ 3.0
            N = 1,000 → E[SR_max] ≈ 3.7
            N = 10,000 → E[SR_max] ≈ 4.3
            N = 100,000 → E[SR_max] ≈ 4.8
        """
        if n_trials <= 1:
            return 0.0
        return np.sqrt(2 * np.log(n_trials))


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    T: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Convenience function to calculate DSR probability.

    Args:
        sharpe: Observed annualized Sharpe ratio
        n_trials: Number of strategies tested
        T: Sample size (number of observations)
        skewness: Skewness of returns
        kurtosis: Kurtosis of returns

    Returns:
        DSR probability (0 to 1)
    """
    dsr = DeflatedSharpeRatio()
    result = dsr.calculate(sharpe, n_trials, T, skewness, kurtosis)
    return result.dsr_pvalue
