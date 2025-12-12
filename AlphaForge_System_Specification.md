# AlphaForge v2.0 System Specification

**Version:** 2.0.1
**Date:** December 12, 2025
**Status:** Design Phase - Specification Complete

---

## 1. Executive Summary

### 1.1 Vision

AlphaForge is a production-grade platform for systematic trading strategy discovery, validation, and deployment. It implements **defense-in-depth** against the primary failure modes in quantitative finance: overfitting, data snooping, and execution reality mismatch.

### 1.2 Core Value Proposition

| Problem | Industry Failure Rate | AlphaForge Solution |
|---------|----------------------|---------------------|
| False discovery (overfitting) | 95% of published factors | CPCV + PBO + DSR validation |
| Lookahead bias | 60-80% of retail algos | Bi-temporal data architecture |
| Execution gap | 70% fail within 6 months | Shadow trading + realistic impact modeling |

### 1.3 Key Differentiators

| Capability | Traditional Platforms | AlphaForge |
|------------|----------------------|------------|
| Cross-Validation | Standard K-Fold (leaky) | Combinatorially Purged CV |
| Overfitting Detection | Train/test split only | PBO < 0.05, DSR > 0.95 |
| Multiple Testing | None | Deflated Sharpe at Stage 1 |
| Data Integrity | Single timestamp | Bi-temporal (3 timestamps) |
| Execution Modeling | Constant slippage | Almgren-Chriss + queue simulation |
| Paper Trading | Delayed feeds | Shadow trading with live order books |
| Monitoring | Manual review | Real-time CUSUM/SPRT |
| Stress Testing | Historical replay | Historical + synthetic shocks |

### 1.4 Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| PBO | < 0.05 (deploy), < 0.02 (auto-accept) | Probability of backtest overfitting |
| DSR | > 0.95 | 95% confidence of positive Sharpe |
| Implementation Shortfall | < 20% | Backtest vs live performance gap |
| Detection Time | < 5 days | Mean time to detect strategy degradation |
| Funnel Pass Rate | ~0.1% | 10,000 candidates → 2-3 deployed |

### 1.5 Timeline & Investment

| Phase | Duration | Cost |
|-------|----------|------|
| Specification | Complete | - |
| Implementation | 20-24 weeks | $35-45K infrastructure |
| Team (5 FTEs) | Ongoing | $500K-1M/year |
| **Total to Production** | **24-28 weeks** | **~$600K Year 1** |

---

## 2. Architecture Overview

### 2.1 Seven-Layer Defense Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 7: GOVERNANCE & MONITORING                                │
│ MLflow | Prometheus | CUSUM/SPRT | SHAP Explainability         │
├─────────────────────────────────────────────────────────────────┤
│ Layer 6: PRODUCTION EXECUTION                                   │
│ Live Trading | Shadow Trading | Paper Trading                   │
│ Almgren-Chriss Impact | Queue Simulation | FIX Protocol        │
├─────────────────────────────────────────────────────────────────┤
│ Layer 5: STATISTICAL VALIDATION                                 │
│ CPCV | PBO Calculator | DSR | arch.SPA | Stress Testing        │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: HYBRID BACKTESTING                                     │
│ Vectorized (vectorbt) | Event-Driven (NautilusTrader)          │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: STRATEGY FACTORY                                       │
│ Genetic Programming (DEAP) | Bayesian Optimization (Optuna)    │
│ Foundation Models (Chronos-2) | LLM Ideas (with temporal safety)│
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: POINT-IN-TIME FEATURE STORE                           │
│ Feast | Technical Indicators | Macro (ALFRED) | Alternative    │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: BI-TEMPORAL DATA LAKE                                  │
│ Hot Store (Arrow/NVMe) | Cold Store (Iceberg/MinIO)            │
│ Kafka Streaming | ALFRED Vintages | Corporate Actions          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Strategy Validation Funnel

```
10,000 Strategy Candidates
    │
    ▼ Stage 1: Vectorized Screening (DSR > 0.95)
    │ [Apply deflated Sharpe accounting for ALL 10K trials]
    │
~50-100 Strategies (0.5-1%)
    │
    ▼ Stage 2: CPCV + PBO Validation
    │ [16 blocks, C(16,8)=12,870 combinations, PBO < 0.05]
    │
~20 Strategies
    │
    ▼ Stage 3: Event-Driven Backtest
    │ [NautilusTrader, realistic execution, shortfall < 30%]
    │
~10 Strategies
    │
    ▼ Stage 4: Multiple Testing + Stress
    │ [arch.SPA test, historical + synthetic stress scenarios]
    │
~5 Finalists
    │
    ▼ Stage 5: Shadow Trading (30-365 days by type)
    │ [Live order book simulation, divergence < 20%]
    │
2-3 Deployed Strategies
```

### 2.3 Technology Stack

```yaml
Languages:
  Python: 85% (data, ML, backtesting, validation)
  Rust: 10% (NautilusTrader core)
  SQL/YAML: 5% (queries, configs)

Data_Layer:
  Cold_Store: Apache Iceberg on MinIO/S3
  Hot_Store: Arrow IPC / Feather on NVMe
  Streaming: Kafka / Redpanda
  Metadata: PostgreSQL 15+
  Macro_Vintages: ALFRED API (Federal Reserve)

Feature_Store:
  Core: Feast 0.35+
  Compute: Polars 0.19+
  Indicators: TA-Lib

Strategy_Factory:
  Distributed: Ray 2.9+
  Optimization: Optuna 3.5+
  Genetic: DEAP
  Foundation_Models: Chronos-2, TimesMoE

Backtesting:
  Vectorized: vectorbt PRO (or OSS fallback)
  Event_Driven: NautilusTrader 1.190+
  Market_Impact: Almgren-Chriss (parametric)

Validation:
  Statistics: SciPy, statsmodels
  Bootstrap: arch 6.0+ (SPA, StepM)
  Custom: CPCV, PBO, DSR implementations

Execution:
  Core: NautilusTrader (same as backtest)
  Protocol: FIX via quickfix
  Shadow: Custom live book simulator

Monitoring:
  Experiments: MLflow 2.10+
  Metrics: Prometheus 2.48+
  Dashboards: Grafana 10.2+
  Explainability: SHAP 0.44+

Infrastructure:
  Orchestration: Kubernetes 1.28+
  CI/CD: GitLab CI / GitHub Actions
  Containers: Docker
```

---

## 3. Layer 1: Bi-Temporal Data Lake

### 3.1 Purpose

Eliminate lookahead bias at the data infrastructure level through bi-temporal design and Hot/Cold architecture for performance.

### 3.2 Three-Timestamp Schema

Every data point carries three timestamps:

| Timestamp | Meaning | Example |
|-----------|---------|---------|
| `observation_date` | What period the data measures | Q1 2020 |
| `release_date` | When it became publicly available | 2020-04-29 |
| `transaction_time` | When entered our database | 2020-04-29T08:31:23 |

```sql
CREATE TABLE macro_indicators (
    indicator_name VARCHAR(100),
    country VARCHAR(3),
    observation_date DATE,
    release_date TIMESTAMP,
    transaction_time TIMESTAMP,
    valid_to TIMESTAMP,  -- NULL if current version
    value DECIMAL(20,6),
    revision_number INT,
    source VARCHAR(100),
    PRIMARY KEY (indicator_name, country, observation_date, release_date)
);
```

### 3.3 Point-in-Time Query

```python
def get_pit_value(indicator: str, obs_date: date, as_of: datetime) -> float:
    """Returns value AS IT WAS KNOWN at as_of timestamp."""
    return db.query("""
        SELECT value FROM macro_indicators
        WHERE indicator_name = :indicator
          AND observation_date = :obs_date
          AND release_date <= :as_of
          AND (valid_to > :as_of OR valid_to IS NULL)
        ORDER BY release_date DESC LIMIT 1
    """, indicator=indicator, obs_date=obs_date, as_of=as_of)
```

### 3.4 Hot/Cold Architecture

```yaml
Cold_Store:
  Technology: Apache Iceberg on MinIO
  Purpose: Source of truth, audit, compliance, time-travel
  Write: Real-time Kafka streaming
  Read: Feature engineering, EOD processes, regulatory queries
  Retention: Indefinite (immutable)

Hot_Store:
  Technology: Arrow IPC / Feather on NVMe SSD
  Purpose: High-performance backtesting
  Refresh: Daily at 4:30 PM ET (post-market)
  Read: Vectorized screening (10M+ bars/sec)
  Retention: 7-day rolling cache

Cache_Builder:
  Schedule: Daily 4:30 PM ET
  Process:
    1. Query Iceberg for all symbols (10 years)
    2. Export to Arrow format
    3. Distribute to compute node NVMe
    4. Validate checksums
    5. Alert on failure
```

### 3.5 ALFRED Integration

Federal Reserve vintage data for macro indicators:

```python
MACRO_SERIES = {
    'GDP': 'Gross Domestic Product',
    'UNRATE': 'Unemployment Rate',
    'CPIAUCSL': 'Consumer Price Index',
    'FEDFUNDS': 'Federal Funds Rate',
    'DGS10': '10-Year Treasury',
    'T10Y2Y': 'Yield Curve Spread',
    'UMCSENT': 'Consumer Sentiment',
    # ... 20+ indicators with full vintage history
}

# Daily sync at 10 AM ET (after most releases)
def daily_alfred_sync():
    for series_id in MACRO_SERIES:
        vintage = alfred.get_observations(series_id, vintage_date=today)
        upsert_to_iceberg(vintage)
```

### 3.6 Survivorship Bias Prevention

```python
def validate_universe(universe: str, start_date: date) -> bool:
    """Ensures backtest includes delisted securities."""
    historical = get_historical_constituents(universe, start_date)
    available = get_available_symbols()
    missing = historical - available

    if missing:
        raise SurvivorshipBiasError(f"Missing {len(missing)} delisted symbols")
    return True

# Required data sources (choose one):
# - CRSP (academic gold standard)
# - Sharadar Core US Equities (Nasdaq Data Link)
# - Norgate Data (retail-accessible)
```

### 3.7 Data Quality Checks

| Check | Description | Frequency |
|-------|-------------|-----------|
| Duplicate timestamps | No duplicate (symbol, timestamp) | Daily |
| Temporal ordering | ingestion_time >= data timestamp | Daily |
| OHLC consistency | high >= low, high >= open/close | Daily |
| Missing data | Gap detection with exchange calendar | Daily |
| Price sanity | Flag >50% moves (check for splits) | Daily |
| Survivorship | Universe includes delisted | Per backtest |

---

## 4. Layer 2: Point-in-Time Feature Store

### 4.1 Purpose

Ensure all features are computed using only information available at query time.

### 4.2 Feature Views (Logical Separation)

```yaml
Technical_Indicators:
  Entity: symbol
  Timestamp: trade_timestamp (simple)
  Source: Hot Store (Arrow) + Redis cache
  Features:
    - RSI(14), RSI(7)
    - MACD, MACD_signal, MACD_hist
    - SMA(20), SMA(50), SMA(200)
    - Bollinger bands (20, 2)
    - ATR(14)
    - Volume SMA ratio
  Refresh: Real-time (streaming)

Macro_Fundamentals:
  Entity: country/market
  Timestamp: release_date (bi-temporal)
  Source: Cold Store (Iceberg)
  Features:
    - GDP growth (latest vintage)
    - Unemployment rate
    - CPI YoY
    - Fed Funds rate
    - Yield curve slope
  Refresh: Daily (after ALFRED sync)

Alternative_Data:
  Entity: symbol
  Timestamp: publication_timestamp
  Source: PostgreSQL
  Features:
    - LLM sentiment (with temporal safety)
    - News volume
    - Social mentions
  Refresh: Hourly
```

### 4.3 LLM Temporal Safety

```python
class LLMLeakageDetector:
    """Tests LLM for future knowledge contamination."""

    CANARY_QUESTIONS = [
        {'question': 'Who won 2020 US election?',
         'earliest_knowable': '2020-11-07'},
        {'question': 'S&P 500 2020 return?',
         'earliest_knowable': '2021-01-01'},
        {'question': 'When was COVID vaccine approved?',
         'earliest_knowable': '2020-12-11'},
    ]

    def test_isolation(self, llm, cutoff_date: datetime) -> bool:
        """Returns True if LLM passes temporal isolation."""
        for canary in self.CANARY_QUESTIONS:
            if canary['earliest_knowable'] > cutoff_date.isoformat():
                response = llm.query(canary['question'])
                if self._has_future_knowledge(response, canary):
                    raise TemporalLeakageError(canary['question'])
        return True

# REQUIRED: Run before any LLM feature deployment
```

### 4.4 Feature Engineering Rules

```python
# FORBIDDEN: Full-sample statistics
df['z_score'] = (df['price'] - df['price'].mean()) / df['price'].std()

# REQUIRED: Rolling/expanding only
df['z_score'] = df['price'].rolling(252).apply(
    lambda x: (x.iloc[-1] - x.mean()) / x.std()
)

# FORBIDDEN: Centered windows
df['ma'] = df['price'].rolling(20, center=True).mean()

# REQUIRED: Trailing windows only
df['ma'] = df['price'].rolling(20, center=False).mean()
```

---

## 5. Layer 3: Strategy Factory

### 5.1 Purpose

Generate diverse strategy candidates through multiple approaches.

### 5.2 Strategy Genome

All strategies are serialized to a standard JSON format:

```python
@dataclass
class StrategyGenome:
    """Universal strategy representation."""

    # Identity
    id: str
    name: str
    version: str

    # Signals
    entry_rules: List[Rule]      # When to enter
    exit_rules: List[Rule]       # When to exit

    # Filters
    universe_filter: Filter      # Which assets
    regime_filter: Filter        # Market conditions

    # Position sizing
    sizing_method: str           # 'fixed', 'volatility', 'kelly'
    max_position_pct: float

    # Execution
    order_type: str              # 'market', 'limit', 'vwap'
    urgency: str                 # 'passive', 'neutral', 'aggressive'

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyGenome':
        return cls(**json.loads(json_str))
```

### 5.3 Generation Methods

```yaml
Genetic_Programming:
  Library: DEAP
  Population: 1000
  Generations: 50
  Operations:
    - Mutation (parameter tweaks)
    - Crossover (combine strategies)
    - Selection (tournament, top performers)
  Output: ~5000 variants per run

Bayesian_Optimization:
  Library: Optuna + Ray Tune
  Trials: 500-1000 per strategy template
  Hyperparameters:
    - Lookback periods
    - Threshold values
    - Indicator parameters
  Output: ~1000 optimized variants

Foundation_Models:
  Models: Chronos-2, TimesMoE
  Use: Zero-shot forecasting signals
  Integration: Feature in strategy rules
  Output: ~100 model-based strategies

LLM_Ideation:
  Model: GPT-4 / FinGPT
  Use: Generate strategy concepts
  Safety: Temporal isolation required
  Output: ~100 novel ideas → variants
```

---

## 6. Layer 4: Hybrid Backtesting Engine

### 6.1 Purpose

Two-stage backtesting: fast screening then high-fidelity validation.

### 6.2 Vectorized Screening (Stage 1)

```python
# Configuration
ENGINE_OPTIONS = {
    'vectorbt_pro': {
        'speed': '10M+ bars/sec',
        'license': '$300/year',
        'recommended_for': 'production'
    },
    'vectorbt_oss': {
        'speed': '2M+ bars/sec',
        'license': 'Apache 2.0 (free)',
        'recommended_for': 'research'
    },
    'numpy_custom': {
        'speed': '1-5M bars/sec',
        'license': 'MIT (free)',
        'recommended_for': 'academic'
    }
}

def vectorized_screen(strategies: List[StrategyGenome],
                      data: pa.Table,
                      n_trials: int) -> List[Dict]:
    """
    Fast screening with multiple testing correction.

    CRITICAL: Apply DSR at this stage to account for all trials.
    """
    results = []

    for strategy in strategies:
        metrics = vectorbt_backtest(strategy, data)

        # Deflate Sharpe for multiple testing
        dsr = deflated_sharpe_ratio(
            sharpe=metrics['sharpe'],
            n_trials=n_trials,  # Total strategies tested
            T=len(data)
        )

        results.append({
            'strategy': strategy,
            'sharpe': metrics['sharpe'],
            'dsr': dsr,
            'passed': dsr > 0.95
        })

    return [r for r in results if r['passed']]
```

### 6.3 Event-Driven Validation (Stage 3)

```python
# NautilusTrader for high-fidelity simulation
from nautilus_trader.backtest import BacktestEngine

def event_driven_backtest(strategy: StrategyGenome,
                          data: DataCatalog) -> BacktestResult:
    """
    Full execution simulation with realistic costs.
    """
    engine = BacktestEngine()

    # Configure execution model
    engine.add_execution_model(
        market_impact=AlmgrenChrissModel(
            permanent_impact=0.314,
            temporary_impact=0.142
        ),
        queue_position=EmpiricalQueueModel(),
        partial_fills=True,
        latency_ms=50
    )

    # Run backtest
    result = engine.run(strategy, data)

    # Calculate implementation shortfall
    shortfall = (result.backtest_sharpe - result.realistic_sharpe) / result.backtest_sharpe

    return BacktestResult(
        strategy=strategy,
        sharpe=result.realistic_sharpe,
        shortfall=shortfall,
        passed=shortfall < 0.30
    )
```

### 6.4 Market Impact Model (Almgren-Chriss)

```python
def market_impact(order_size: float,
                  daily_volume: float,
                  volatility: float,
                  time_horizon: float = 1.0) -> float:
    """
    Almgren-Chriss parametric market impact model.
    Industry standard baseline for v2.0.

    ML model planned for v2.1 after collecting internal execution data.
    """
    participation = order_size / daily_volume

    # Permanent impact (information leakage)
    permanent = 0.314 * (volatility ** 0.5) * participation

    # Temporary impact (execution pressure)
    temporary = 0.142 * (volatility ** 0.5) * (participation ** 0.6) / time_horizon

    return permanent + temporary
```

---

## 7. Layer 5: Statistical Validation Pipeline

### 7.1 Purpose

Rigorous statistical testing to eliminate false discoveries.

### 7.2 Validation Threshold Policy

| Metric | Minimum (Deploy) | Default (Auto-Accept) | Crisis Regime |
|--------|------------------|----------------------|---------------|
| PBO | < 0.05 | < 0.02 | < 0.01 |
| DSR | > 0.95 | > 0.98 | > 0.99 |
| Sharpe | > 1.0 | > 1.5 | > 2.0 |

**Policy:**
- **Minimum**: Strategy MAY deploy with committee approval
- **Default**: Strategy AUTO-ACCEPTS (no manual review)
- **Crisis**: Tightened thresholds during high-volatility regimes

### 7.3 CPCV Implementation

```python
def combinatorial_purged_cv(strategy: StrategyGenome,
                            data: pd.DataFrame,
                            n_splits: int = 16,
                            test_splits: int = 8,
                            embargo_pct: float = 0.02) -> CPCVResult:
    """
    Combinatorially Purged Cross-Validation.
    Tests all C(16,8) = 12,870 train/test combinations.
    """
    from itertools import combinations

    # Create time-based splits
    splits = np.array_split(data, n_splits)

    results = []

    # Test all combinations
    for test_indices in combinations(range(n_splits), test_splits):
        train_indices = [i for i in range(n_splits) if i not in test_indices]

        # Purge: remove data near boundaries
        train_data = purge_data(splits, train_indices, embargo_pct)
        test_data = concat_splits(splits, test_indices)

        # Train and test
        metrics = backtest_on_split(strategy, train_data, test_data)
        results.append(metrics['sharpe'])

    # Calculate PBO
    n_negative = sum(1 for r in results if r < 0)
    pbo = n_negative / len(results)

    return CPCVResult(
        sharpe_distribution=results,
        pbo=pbo,
        passed=pbo < 0.05
    )
```

### 7.4 Hansen's SPA Test (via arch library)

```python
from arch.bootstrap import SPA

def superior_predictive_ability(strategy_returns: np.array,
                                benchmark_returns: np.array,
                                block_size: int = 10) -> SPAResult:
    """
    Hansen's Superior Predictive Ability test.
    Uses arch library (production-tested, NOT custom implementation).
    """
    # Calculate losses (negative returns for SPA)
    strategy_losses = -strategy_returns
    benchmark_losses = -benchmark_returns

    # Run SPA test
    spa = SPA(
        benchmark_losses,
        strategy_losses.reshape(-1, 1),
        block_size=block_size,
        bootstrap='stationary'
    )
    result = spa.compute()

    return SPAResult(
        pvalue=result.pvalue[0],
        passed=result.pvalue[0] < 0.05
    )
```

### 7.5 Deflated Sharpe Ratio

```python
def deflated_sharpe_ratio(sharpe: float,
                          n_trials: int,
                          T: int,
                          skewness: float = 0,
                          kurtosis: float = 3) -> float:
    """
    DSR accounts for multiple testing, non-normality, and autocorrelation.
    From Bailey & Lopez de Prado (2014).
    """
    from scipy.stats import norm

    # Expected maximum Sharpe under null
    e_max_sharpe = expected_max_sharpe(n_trials, T)

    # Standard error of Sharpe
    se_sharpe = np.sqrt(
        (1 + 0.5 * sharpe**2 - skewness * sharpe + (kurtosis - 3) / 4 * sharpe**2) / T
    )

    # Probability that observed Sharpe exceeds expected max noise
    dsr = norm.cdf((sharpe - e_max_sharpe) / se_sharpe)

    return dsr

def expected_max_sharpe(n_trials: int, T: int) -> float:
    """Expected maximum Sharpe from N random strategies."""
    from scipy.stats import norm

    gamma = 0.5772  # Euler-Mascheroni constant
    return norm.ppf(1 - 1/n_trials) * (1 - gamma) + gamma * norm.ppf(1 - 1/(n_trials * np.e))
```

### 7.6 Regime-Adaptive Validation

```python
class RegimeDetector:
    """
    Point-in-time regime classification.
    CRITICAL: Uses only data available at detection time.
    """

    def __init__(self, model_version: str):
        self.model = load_frozen_model(model_version)
        self.version = model_version

    def detect(self, as_of_date: datetime) -> str:
        """Detect regime using ONLY data up to as_of_date."""
        features = {
            'vix': get_pit_value('VIX', as_of_date),
            'realized_vol': calc_realized_vol(as_of_date, lookback=20),
            'drawdown': calc_max_drawdown(as_of_date, lookback=60)
        }

        regime = self.model.predict(features)

        # Audit log
        log_regime_decision(as_of_date, regime, self.version, features)

        return regime  # 'normal', 'high_vol', 'crisis', 'trending'

    def get_thresholds(self, regime: str) -> Dict:
        """Return validation thresholds for regime."""
        THRESHOLDS = {
            'normal':   {'pbo': 0.05, 'dsr': 0.95, 'sharpe': 1.0},
            'trending': {'pbo': 0.07, 'dsr': 0.93, 'sharpe': 0.8},
            'high_vol': {'pbo': 0.03, 'dsr': 0.97, 'sharpe': 1.2},
            'crisis':   {'pbo': 0.02, 'dsr': 0.99, 'sharpe': 2.0},
        }
        return THRESHOLDS[regime]
```

### 7.7 Stress Testing

```python
STRESS_SCENARIOS = {
    # Historical replays
    '2008_financial_crisis': {
        'period': ('2008-09-01', '2008-12-31'),
        'description': 'Lehman collapse, credit freeze'
    },
    '2020_covid_crash': {
        'period': ('2020-02-19', '2020-03-23'),
        'description': 'Fastest 30% drop in history'
    },
    '2022_rate_hikes': {
        'period': ('2022-01-01', '2022-10-31'),
        'description': 'Bond-equity correlation breakdown'
    },

    # Synthetic shocks
    'correlation_spike': {
        'transform': lambda returns: apply_correlation(returns, rho=0.95),
        'description': 'All assets correlate (crisis behavior)'
    },
    'volatility_3x': {
        'transform': lambda returns: returns * 3,
        'description': 'Triple normal volatility'
    },
    'liquidity_drain': {
        'impact_multiplier': 5.0,
        'description': 'Spreads widen 5x, depth drops 80%'
    },
}

def run_stress_tests(strategy: StrategyGenome) -> StressResult:
    """Run all stress scenarios, require 80% pass rate."""
    results = []

    for name, scenario in STRESS_SCENARIOS.items():
        if 'period' in scenario:
            result = backtest_period(strategy, *scenario['period'])
        else:
            result = backtest_transformed(strategy, scenario['transform'])

        passed = result.sharpe > 0 and result.max_drawdown < 0.50
        results.append({'scenario': name, 'passed': passed})

    pass_rate = sum(r['passed'] for r in results) / len(results)

    return StressResult(results=results, pass_rate=pass_rate, passed=pass_rate >= 0.80)
```

---

## 8. Layer 6: Production Execution

### 8.1 Purpose

Seamless transition from backtest to live with shadow trading validation.

### 8.2 Execution Modes

```yaml
Shadow_Trading:
  Description: Run on live data, simulate fills, no real orders
  Duration: 30-365 days (by strategy type)
  Validation: Compare to backtest expectations
  Graduation: Divergence < 20%

Paper_Trading:
  Description: Delayed data feed, simulated fills
  Use_Case: Initial testing, educational
  Duration: Optional (no minimum)

Live_Trading:
  Description: Real orders, real money
  Ramp: Start 10% allocation, increase weekly
  Monitoring: Real-time CUSUM/SPRT
```

### 8.3 Shadow Trading Duration by Strategy Type

```python
SHADOW_REQUIREMENTS = {
    'intraday':   {'min_days': 30,  'rationale': 'Many observations per day'},
    'daily':      {'min_days': 90,  'rationale': 'Standard daily strategies'},
    'swing':      {'min_days': 120, 'rationale': 'Multi-day holds'},
    'quarterly':  {'min_days': 180, 'rationale': 'Cover 2 earnings cycles'},
    'seasonal':   {'min_days': 365, 'rationale': 'Full annual cycle'},
}

def get_shadow_duration(strategy: StrategyGenome) -> int:
    """Returns minimum shadow trading days for strategy type."""
    return SHADOW_REQUIREMENTS[strategy.frequency]['min_days']
```

### 8.4 Shadow Engine with Phantom Liquidity Handling

```python
class ShadowTradingEngine:
    """
    Simulates order execution against live order book.
    Tracks virtual liquidity consumption to avoid double-counting.
    """

    def __init__(self):
        self.virtual_book = {}  # Tracks consumed liquidity
        self.strategies = []

    def simulate_fill(self, strategy_id: str,
                      order: Order,
                      live_book: OrderBook) -> Fill:
        """
        Simulate fill accounting for virtual consumption.
        """
        symbol = order.symbol

        # Get available liquidity (live - already consumed)
        if symbol not in self.virtual_book:
            self.virtual_book[symbol] = {}

        available = live_book.get_depth(order.side) - self.virtual_book[symbol].get(order.price, 0)

        # Calculate fill
        fill_qty = min(order.quantity, available * 0.8)  # 80% haircut
        fill_price = self._estimate_fill_price(order, live_book, fill_qty)

        # Update virtual consumption
        self.virtual_book[symbol][order.price] = \
            self.virtual_book[symbol].get(order.price, 0) + fill_qty

        return Fill(
            strategy_id=strategy_id,
            quantity=fill_qty,
            price=fill_price,
            slippage=fill_price - order.price
        )

    def reset_virtual_book(self):
        """Called periodically to refresh from live book."""
        self.virtual_book = {}
```

### 8.5 Live Trading Deployment

```python
class LiveDeployment:
    """Manages strategy deployment lifecycle."""

    RAMP_SCHEDULE = [0.10, 0.25, 0.50, 0.75, 1.00]  # Weekly ramp

    def deploy(self, strategy: StrategyGenome, capital: float):
        """Deploy with gradual capital ramp."""

        # Initial allocation (10%)
        allocation = capital * self.RAMP_SCHEDULE[0]

        # Create NautilusTrader instance
        trader = NautilusTrader(
            strategy=strategy,
            execution_model=AlmgrenChrissModel(),
            capital=allocation
        )

        # Register with monitoring
        monitor.register(strategy.id, expected_sharpe=strategy.backtest_sharpe)

        # Schedule ramp increases
        for week, pct in enumerate(self.RAMP_SCHEDULE[1:], 1):
            scheduler.schedule(
                days=week * 7,
                action=lambda: self._increase_allocation(strategy.id, capital * pct)
            )

        return trader
```

---

## 9. Layer 7: Governance & Monitoring

### 9.1 Purpose

Real-time performance monitoring with automated intervention.

### 9.2 Strategy State Machine

```python
class StrategyState(Enum):
    ACTIVE = "active"           # Normal operation
    WATCHING = "watching"       # Warning threshold breached
    DEGRADED = "degraded"       # Performance threshold breached
    PAUSED = "paused"          # Halted, investigation pending
    RETIRED = "retired"        # Permanently shut down

STATE_TRANSITIONS = {
    'ACTIVE -> WATCHING': {
        'trigger': 'cusum > 3.0 OR drawdown > 15%',
        'action': 'alert_only',
        'review_required': False
    },
    'WATCHING -> DEGRADED': {
        'trigger': 'cusum > 5.0 OR drawdown > 25%',
        'action': 'reduce_allocation_50%',
        'review_required': True,
        'review_deadline': '24 hours'
    },
    'DEGRADED -> PAUSED': {
        'trigger': 'cusum > 7.0 OR drawdown > 35%',
        'action': 'halt_trading',
        'review_required': True,
        'review_deadline': '4 hours'
    },
    'PAUSED -> RETIRED': {
        'trigger': 'committee_vote = shutdown',
        'action': 'unwind_positions',
        'unwind_horizon': '5 days'
    },
    'PAUSED -> ACTIVE': {
        'trigger': 'investigation_complete AND fix_implemented',
        'action': 'gradual_ramp',
        'ramp_schedule': [0.25, 0.50, 0.75, 1.00]
    }
}
```

### 9.3 Probabilistic CUSUM Monitoring

```python
class ProbabilisticCUSUM:
    """
    Monitors Information Ratio (not raw P&L).
    Reduces false alarm rate vs naive CUSUM.
    """

    def __init__(self, expected_ir: float, k: float = 0.5, h: float = 5.0):
        self.expected_ir = expected_ir
        self.k = k      # Drift allowance (std devs)
        self.h = h      # Alarm threshold
        self.S = 0.0    # CUSUM statistic

    def update(self, realized_returns: np.array,
               benchmark_returns: np.array) -> bool:
        """
        Update CUSUM and return True if alarm triggered.
        """
        # Calculate rolling Information Ratio
        excess = realized_returns - benchmark_returns
        realized_ir = excess.mean() / excess.std() * np.sqrt(252)

        # Standardize deviation from expected
        z = (realized_ir - self.expected_ir) / self._stderr()

        # Update CUSUM (one-sided, detecting degradation)
        self.S = max(0, self.S + (-z) - self.k)

        return self.S > self.h

    def reset(self):
        self.S = 0.0
```

### 9.4 Experiment Tracking (MLflow)

```python
# Every strategy evaluation logged
with mlflow.start_run(run_name=f"strategy_{strategy.id}"):
    mlflow.log_params({
        'strategy_id': strategy.id,
        'strategy_version': strategy.version,
        'data_snapshot': iceberg_snapshot_id,
        'code_commit': git_commit_sha,
        'random_seed': seed
    })

    mlflow.log_metrics({
        'sharpe': result.sharpe,
        'pbo': result.pbo,
        'dsr': result.dsr,
        'max_drawdown': result.max_drawdown,
        'implementation_shortfall': result.shortfall
    })

    mlflow.log_artifact(result.report_path)
```

### 9.5 SHAP Explainability

```python
def explain_decision(strategy: StrategyGenome,
                     decision: TradeDecision,
                     features: Dict) -> Explanation:
    """
    Generate SHAP explanation for regulatory compliance.
    """
    explainer = shap.TreeExplainer(strategy.model)
    shap_values = explainer.shap_values(features)

    # Generate natural language explanation
    top_factors = get_top_factors(shap_values, n=3)

    explanation = f"""
    Trade Decision: {decision.action} {decision.quantity} {decision.symbol}

    Key Factors:
    1. {top_factors[0].name}: {top_factors[0].contribution:+.2%}
    2. {top_factors[1].name}: {top_factors[1].contribution:+.2%}
    3. {top_factors[2].name}: {top_factors[2].contribution:+.2%}

    Confidence: {decision.confidence:.1%}
    """

    # Store for audit trail
    store_explanation(strategy.id, decision.id, explanation, shap_values)

    return Explanation(text=explanation, shap_values=shap_values)
```

---

## 10. Risk Management Framework

### 10.1 Position Sizing

```python
POSITION_SIZING = {
    'method': 'volatility_targeting',
    'target_volatility': 0.10,    # 10% annualized per strategy
    'max_leverage': 1.0,          # No leverage initially
    'min_position': 0.01,         # 1% minimum
    'max_position': 0.20,         # 20% maximum per strategy
}

def calculate_position_size(strategy: StrategyGenome,
                            portfolio_value: float,
                            current_vol: float) -> float:
    """Volatility-targeted position sizing."""
    target_vol = POSITION_SIZING['target_volatility']
    raw_size = (target_vol / current_vol) * portfolio_value

    # Apply limits
    min_size = portfolio_value * POSITION_SIZING['min_position']
    max_size = portfolio_value * POSITION_SIZING['max_position']

    return np.clip(raw_size, min_size, max_size)
```

### 10.2 Strategy-Level Limits

```python
STRATEGY_LIMITS = {
    'max_position_pct': 0.20,      # 20% of portfolio per strategy
    'max_daily_loss': 0.02,        # 2% daily stop-loss
    'max_drawdown': 0.15,          # 15% triggers review
    'max_drawdown_hard': 0.20,     # 20% triggers retirement
    'max_consecutive_losses': 10,  # Days
}
```

### 10.3 Portfolio-Level Limits

```python
PORTFOLIO_LIMITS = {
    'max_correlation': 0.70,       # Max pairwise strategy correlation
    'var_95': 0.03,                # 95% VaR < 3% daily
    'cvar_95': 0.05,               # 95% CVaR < 5% daily
    'max_sector_exposure': 0.40,   # 40% max in any sector
    'max_single_name': 0.10,       # 10% max in any single security
    'max_strategies': 20,          # Maximum concurrent strategies
}

def check_portfolio_risk(portfolio: Portfolio) -> RiskReport:
    """Daily portfolio risk check."""
    violations = []

    # Correlation check
    corr_matrix = portfolio.strategy_correlation_matrix()
    max_corr = corr_matrix.max()
    if max_corr > PORTFOLIO_LIMITS['max_correlation']:
        violations.append(f"Correlation {max_corr:.2f} > {PORTFOLIO_LIMITS['max_correlation']}")

    # VaR check
    var = portfolio.calculate_var(confidence=0.95)
    if var > PORTFOLIO_LIMITS['var_95']:
        violations.append(f"VaR {var:.2%} > {PORTFOLIO_LIMITS['var_95']:.2%}")

    # Sector exposure
    for sector, exposure in portfolio.sector_exposures().items():
        if exposure > PORTFOLIO_LIMITS['max_sector_exposure']:
            violations.append(f"{sector} exposure {exposure:.2%} > limit")

    return RiskReport(
        passed=len(violations) == 0,
        violations=violations,
        var_95=var,
        max_correlation=max_corr
    )
```

### 10.4 Circuit Breakers

```python
CIRCUIT_BREAKERS = {
    'strategy_daily_loss': {
        'threshold': 0.03,
        'action': 'halt_strategy',
        'description': 'Halt if strategy loses 3% in single day'
    },
    'portfolio_daily_loss': {
        'threshold': 0.05,
        'action': 'halt_all',
        'description': 'Halt all strategies if portfolio loses 5%'
    },
    'strategy_drawdown': {
        'threshold': 0.20,
        'action': 'retire_strategy',
        'description': 'Retire strategy if 20% from peak'
    },
    'cusum_breach': {
        'threshold': 7.0,
        'action': 'pause_strategy',
        'description': 'Pause for investigation if CUSUM > 7'
    },
}

def check_circuit_breakers(strategy: Strategy, portfolio: Portfolio):
    """Real-time circuit breaker check."""
    for name, breaker in CIRCUIT_BREAKERS.items():
        if breaker_triggered(strategy, portfolio, breaker):
            execute_action(breaker['action'], strategy)
            send_alert(f"Circuit breaker triggered: {name}")
```

---

## 11. Implementation Roadmap

### 11.1 Phase Overview

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Foundation | Weeks 1-6 | Data Lake + Feature Store |
| Phase 2: Research | Weeks 7-12 | Strategy Factory + Backtesting |
| Phase 3: Validation | Weeks 13-18 | Statistical Pipeline |
| Phase 4: Production | Weeks 19-24 | Execution + Monitoring |
| Phase 5: Hardening | Weeks 25-28 | Testing + Documentation |

### 11.2 Detailed Milestones

```yaml
Week_1-2:
  - Set up Kubernetes cluster
  - Deploy MinIO, PostgreSQL, Kafka
  - Configure Iceberg tables

Week_3-4:
  - ALFRED integration complete
  - Hot/Cold cache builder operational
  - Data quality checks automated

Week_5-6:
  - Feast feature store deployed
  - Technical indicators pipeline
  - LLM temporal safety tests

Week_7-9:
  - Strategy genome schema finalized
  - DEAP genetic programming
  - Optuna integration

Week_10-12:
  - vectorbt integration
  - NautilusTrader backtesting
  - Almgren-Chriss impact model

Week_13-15:
  - CPCV implementation
  - PBO/DSR calculators
  - arch.SPA integration

Week_16-18:
  - Regime detector
  - Stress test framework
  - Validation pipeline end-to-end

Week_19-21:
  - Shadow trading engine
  - Risk management framework
  - Circuit breakers

Week_22-24:
  - CUSUM monitoring
  - MLflow integration
  - Grafana dashboards

Week_25-28:
  - Integration testing
  - Performance optimization
  - Documentation
  - First strategy deployed
```

---

## 12. Future Enhancements (v2.1+)

### 12.1 ML Market Impact Model

```yaml
Target: v2.1 (3 months post-launch)
Approach:
  - Collect 6 months internal execution data
  - Train neural network on actual fills
  - Benchmark vs Almgren-Chriss
  - Deploy if >20% improvement
```

### 12.2 GAN Stress Testing

```yaml
Target: v3.0 (research validation required)
Approach:
  - TimeGAN architecture for financial time series
  - Validate generated scenarios match empirical tail behavior
  - Only deploy after extensive backtesting
Risks:
  - Mode collapse (limited scenario diversity)
  - Missing fat tails
  - Training instability
```

### 12.3 Advanced LLM Integration

```yaml
Target: v2.1
Features:
  - Strategy explanation generation
  - Natural language strategy specification
  - Automated research paper summarization
Requirement: Enhanced temporal safety validation
```

---

## Appendix A: Key Formulas

### A.1 Deflated Sharpe Ratio

```
DSR = Φ((SR - E[SR_max]) / SE[SR])

where:
  SR = observed Sharpe ratio
  E[SR_max] = (1-γ)Φ^(-1)(1-1/N) + γΦ^(-1)(1-1/(Ne))
  SE[SR] = sqrt((1 + 0.5*SR² - skew*SR + (kurt-3)/4*SR²) / T)
  γ = 0.5772 (Euler-Mascheroni constant)
  N = number of trials
  T = sample size
```

### A.2 Expected Maximum Sharpe (Noise)

```
E[SR_max] ≈ sqrt(2 * ln(N))

Example:
  N = 10,000 strategies → E[SR_max] ≈ 4.3
  N = 1,000,000 strategies → E[SR_max] ≈ 5.25
```

### A.3 CUSUM Statistic

```
S_t = max(0, S_{t-1} + (z_t - k))

where:
  z_t = (realized_IR - expected_IR) / stderr(IR)
  k = drift allowance (typically 0.5)

Alarm when: S_t > h (threshold, typically 5.0)
```

### A.4 Almgren-Chriss Market Impact

```
Total Impact = Permanent + Temporary

Permanent = 0.314 * sqrt(σ) * (Q/V)
Temporary = 0.142 * sqrt(σ) * (Q/V)^0.6 / T

where:
  σ = daily volatility
  Q = order size
  V = daily volume
  T = execution horizon (days)
```

---

## Appendix B: Configuration Reference

### B.1 Environment Variables

```bash
# Data Sources
FRED_API_KEY=your_fred_api_key
POLYGON_API_KEY=your_polygon_api_key

# Infrastructure
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
POSTGRES_URL=postgresql://user:pass@postgres:5432/alphaforge
KAFKA_BOOTSTRAP=kafka:9092

# Feature Store
FEAST_REPO_PATH=/app/feature_repo

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Alerts
PAGERDUTY_KEY=your_pagerduty_key
SLACK_WEBHOOK=https://hooks.slack.com/...
```

### B.2 Default Thresholds

```yaml
validation:
  pbo_threshold: 0.02
  dsr_confidence: 0.98
  min_sharpe: 1.5
  max_drawdown: 0.15
  stress_pass_rate: 0.80

execution:
  slippage_bps: 5
  market_impact_model: almgren_chriss
  queue_position_model: empirical

monitoring:
  cusum_k: 0.5
  cusum_h: 5.0
  alert_threshold_yellow: 3.0
  alert_threshold_red: 5.0

risk:
  max_position_pct: 0.20
  max_daily_loss: 0.02
  max_portfolio_var: 0.03
```

---

*Document Version: 2.0.1*
*Last Updated: December 12, 2025*
*Status: Specification Complete - Ready for Implementation*
