# AlphaForge Code & Logic Review: Complete Analysis

## 1. Project Understanding

### Core Architecture

- 5-layer systematic trading platform: Data → Features → Discovery/Strategy → Backtest → Validation
- Multi-objective genetic programming (NSGA-III) for strategy discovery using expression trees with 35+ typed operators
- Statistical validation pipeline: DSR, CPCV, PBO (proxy), SPA, stress testing
- Dual backtest engines: Vectorized (fast screening) + Event-driven (execution realism)
- Real data only: yfinance for OHLCV (adjusted prices), mock/placeholder for FRED and universe constituents
- API-driven architecture: FastAPI backend + Next.js frontend + CLI, with SQLite persistence
- Research-grade implementation: Per `REPORT.md`, many critical fixes already implemented (lookahead bias, minimum activity constraints, persistence, type system split)

### Current State vs. Claims

- "Production-grade" is aspirational—this is a high-quality research prototype
- Strong scientific foundations with documented gaps (true PBO not wired, event-driven engine simplified)
- No live trading infrastructure (deployment is placeholder)

---

## 2. End-to-End Data Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│ ENTRY POINTS                                                         │
│ • CLI (alphaforge.cli)                                              │
│ • REST API (alphaforge.api.server) → FastAPI endpoints              │
│ • Frontend (Next.js) → calls API                                    │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 1: DATA (data/)                                               │
│ MarketDataLoader → yfinance → OHLCVData (pd.DataFrame)             │
│   • Parquet cache (data/cache/) with metadata                      │
│   • Bi-temporal schema (release_timestamp, transaction_timestamp)  │
│   • auto_adjust=True (splits/dividends handled by yfinance)        │
│   • Universe/ALFRED = MOCK DATA (not production-ready)             │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 2: FEATURES (features/)                                       │
│ TechnicalIndicators.compute_all(df) → enriched DataFrame           │
│   • RSI, SMA, EMA, MACD, Bollinger, ATR, ADX, OBV, VWAP            │
│   • All use rolling(..., center=False) for PIT safety              │
│   • Returns computed as close.pct_change()                          │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 3: STRATEGY REPRESENTATION                                    │
│ StrategyGenome (strategy/genome.py)                                 │
│   • Entry/exit rules (RuleGroup)                                    │
│   • Position sizing params                                          │
│   • Stop loss / take profit / max holding days                      │
│                                                                      │
│ OR ExpressionTree (discovery/expression/tree.py)                    │
│   • Root node + typed AST (operators + terminals)                   │
│   • Compiler → pandas/numpy executable                              │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 3.5: DISCOVERY (discovery/)                                   │
│ DiscoveryOrchestrator (orchestrator.py)                            │
│   • NSGA-III evolution on train split (70%)                         │
│   • ExpressionTree → compile → evaluate → signal                   │
│   • Signal → positions via expanding z-score (PIT-safe) ✓          │
│   • Backtest → fitness (Sharpe, DD, Turnover, Complexity)          │
│   • Pareto front → factor zoo (validated on validation split)      │
│   • Returns: DiscoveryResult (pareto_front, best_by_obj, zoo)     │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 4: BACKTESTING (backtest/)                                    │
│ BacktestEngine.run(strategy, data) → BacktestResult                │
│   • Vectorized mode (engine.py):                                    │
│     SignalGenerator → entry/exit signals                            │
│     PositionTracker → positions series (-1/0/1)                     │
│     lagged_positions * price_returns - transaction_costs            │
│     equity_curve = initial_capital * (1+returns).cumprod()          │
│     PerformanceMetrics.from_returns(strategy_returns)               │
│                                                                      │
│   • Event-driven mode (event_driven.py):                            │
│     Order lifecycle (PENDING→FILLED), partial fills                 │
│     Market impact (Almgren-Chriss), latency simulation              │
│     Returns BacktestResult with same structure                      │
│                                                                      │
│ Artifacts:                                                           │
│   • BacktestResult.metrics (Sharpe, returns, DD, trades, etc.)     │
│   • BacktestResult.equity_curve (pd.Series)                         │
│   • BacktestResult.returns (pd.Series)                              │
│   • BacktestResult.positions (pd.Series)                            │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 5: VALIDATION (validation/)                                   │
│ ValidationPipeline.validate(strategy, data, n_trials) →            │
│   ValidationResult                                                   │
│                                                                      │
│ Stage 1: Backtest → BacktestResult                                  │
│ Stage 2: DSR screening (deflated Sharpe w/ multiple testing)       │
│ Stage 3: CPCV (C(16,8)=12,870 splits w/ embargo) → CPCVResult      │
│ Stage 4: PBO proxy (probability of loss) → PBOResult ⚠             │
│ Stage 5: SPA test (optional, vs benchmark) → SPAResult             │
│ Stage 6: Stress testing (optional, 6 scenarios) → StressTestResult │
│                                                                      │
│ Gating logic:                                                        │
│   • passed = DSR≥0.95 AND Sharpe≥1.0 AND (PBO<0.05 OR no CPCV)    │
│   • auto_accept = DSR≥0.98 AND Sharpe≥1.5 AND PBO<0.02            │
│                                                                      │
│ Artifacts:                                                           │
│   • ValidationResult.to_dict() → serialized metrics                 │
│   • ValidationResult.summary() → human-readable report              │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PERSISTENCE & OUTPUT                                                 │
│ • Storage (api/storage.py) → SQLite tables                          │
│   - validation_results (JSON blob)                                  │
│   - factory_results (JSON blob)                                     │
│   - discovery_results (JSON blob)                                   │
│                                                                      │
│ • In-memory caches (server.py globals):                             │
│   - validation_status, factory_status, discovery_status (ephemeral) │
│   - pipeline_stats (not persisted) ⚠                                │
│                                                                      │
│ • Frontend API responses (JSON over HTTP)                           │
│ • CLI output (click.echo, JSON files)                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Objects

- `OHLCVData`: DataFrame + metadata (symbol, timestamps)
- `ExpressionTree`: AST with hash, formula, `complexity_score()`
- `StrategyGenome`: Name, rules, parameters, metadata
- `BacktestResult`: metrics + equity_curve + returns + positions
- `ValidationResult`: passed + auto_accept + component results (DSR/CPCV/PBO/SPA/Stress)

---

## 3. Key Invariants (Platform Must Uphold)

### Point-in-Time Safety (No Lookahead Bias)

- ✓ Feature engineering: All indicators use `rolling(..., center=False)`
- ✓ Signal normalization: Expanding window z-score in `_signals_to_positions()` (verified with regression tests)
- ✓ Expression compiler: Rank/zscore/scale use trailing windows (252-day default)
- ⚠ Data preparation cache: Uses `id(data)` as key—could return stale data if Python reuses IDs (mitigated by weakref)

### Multiple Testing Correction

- ✓ DSR: Adjusts Sharpe threshold based on `n_trials` parameter
- ⚠ Factory pipeline: Calls `validate()` with `n_trials=len(pool.strategies)` correctly
- ⚠ Discovery pipeline: Does NOT call `ValidationPipeline`—no DSR correction for discovered strategies

### Temporal Embargo (Train/Test Contamination)

- ✓ CPCV: Implements 2% embargo between splits
- ✓ Discovery: 70/30 train/validation split (temporal, not random)
- ⚠ Walk-forward: Module exists but not integrated into main pipeline

### Realistic Execution Costs

- ✓ Vectorized engine: 0.1% commission + 0.05% slippage per trade
- ✓ Event-driven engine: Almgren-Chriss market impact model
- ⚠ Partial fills: Modeled but simplified (no order book depth)

### Reproducibility

- ✓ Discovery: Accepts seed parameter, passes to NSGA-III and TreeGenerator
- ⚠ Stress tests: Use `random.Random()` without seeding (non-deterministic)
- ⚠ Factory: Seed not explicitly controlled

### Data Integrity

- ✓ Splits/dividends: `auto_adjust=True` in yfinance fetch
- ⚠ Survivorship bias: `UniverseRegistry` is mock data only
- ⚠ Corporate actions: Relies entirely on yfinance quality (no validation)

---

## 4. Findings (By Severity)

### CRITICAL SEVERITY

#### C1. PBO Mislabeling Risk (Partially Mitigated)

**Title:** Probability of Loss masquerading as PBO could mislead users

**Why it matters:** True PBO measures selection bias across multiple strategies; the current metric measures single-strategy failure rate. Claiming PBO compliance when using a proxy is scientifically invalid.

**Evidence:**

- `src/alphaforge/validation/cpcv.py:150-152` — calculates `pbo = n_negative / len(sharpe_distribution)` (probability of loss)
- `src/alphaforge/validation/pbo.py:178-218` — `calculate_probability_of_loss()` docstring explicitly states “This is NOT true PBO”
- `src/alphaforge/validation/pipeline.py:282` — pipeline uses `calculate_probability_of_loss()`, NOT `ProbabilityOfOverfitting.calculate()`
- True PBO implementation exists (`ProbabilityOfOverfitting.calculate()` lines 69-133) but is never called

**Mitigation status (per `REPORT.md`):**

- Log message at `pipeline.py:282` says “Probability of Loss (single-strategy PBO proxy)”
- Function renamed from generic `calculate_pbo` to explicit `calculate_probability_of_loss`
- Documentation updated to clarify it’s a proxy

**Remaining risk:**

- Frontend/API still labels this as “PBO” in `ValidationResult.to_dict()` and UI
- `README.md` claims “PBO validation” without caveat that it’s a proxy
- Users may not understand the distinction

**How to verify:**

```bash
# Check ValidationResult serialization
grep -n '"pbo"' src/alphaforge/validation/pipeline.py

# Check frontend display
grep -n 'PBO' frontend/components/ValidationRunner.tsx
```

**Suggested fix:**

1. Rename `ValidationResult.pbo_result` → `prob_loss_result`
2. Update API/frontend to display “Probability of Loss” not “PBO”
3. Implement true PBO for discovery pipeline (requires population-level result storage)
4. Add README disclaimer: “PBO proxy (single-strategy robustness) used; true selection-based PBO available but not wired”

---

#### C2. Discovery Pipeline Skips Statistical Validation

**Title:** Discovered strategies bypass DSR/CPCV/SPA validation stages

**Why it matters:** Discovery can produce hundreds of strategies via genetic programming, all tested on same data (massive multiple testing). Without DSR correction, false discoveries are guaranteed.

**Evidence:**

- `src/alphaforge/discovery/orchestrator.py:142-210` — `discover()` method returns `DiscoveryResult` directly
- `src/alphaforge/api/server.py:520-576` — discovery endpoint serializes pareto_front without calling `ValidationPipeline`
- Factory endpoint (lines 325-395) DOES call `pipeline.validate()` with `n_trials=len(pool.strategies)`
- Discovery generates 100-200 strategies × 20-100 generations = 2,000-20,000 trials, but no DSR adjustment

**How to reproduce:**

1. Run discovery with `population_size=100`, `n_generations=20`
2. Observe pareto_front returned with Sharpe ratios
3. No DSR p-value or CPCV robustness check in output
4. Compare to Factory endpoint which calls pipeline.validate()

**Suggested fix:**

```python
# In server.py discovery endpoint, after line 543:
for ind in result.pareto_front:
    validation = pipeline.validate(
        strategy=ind.genome.to_strategy_genome(),
        data=data,
        n_trials=request.population_size * request.n_generations,
        run_cpcv=True,
    )
    # Attach validation.passed flag to ind
```

**Impact:** High risk of overfitting in discovered strategies shipped to users

---

### HIGH SEVERITY

#### H1. Pipeline Statistics Not Persisted

**Title:** pipeline_stats global dict lost on server restart

**Why it matters:** Accumulates “total_generated” and “total_validated” counts for DSR multiple testing correction. If server restarts, DSR will underestimate n_trials.

**Evidence:**

- `src/alphaforge/api/server.py:50-56` — `pipeline_stats = {...}` is in-memory global
- Line 51 comment: `# TODO: Persist stats`
- `server.py:265` and `server.py:563` increment `pipeline_stats["total_generated"]`
- No persistence to storage.db

**How to reproduce:**

1. Run 5 validations → `total_validated = 5`
2. Restart server
3. Run 1 validation → DSR uses `n_trials=1` instead of `n_trials=6`

**Suggested fix:**

```python
# Add to Storage class
def save_pipeline_stats(self, stats: dict):
    with self._get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS pipeline_stats (key TEXT PRIMARY KEY, value INTEGER)")
        for k, v in stats.items():
            conn.execute("INSERT OR REPLACE INTO pipeline_stats VALUES (?, ?)", (k, v))

def load_pipeline_stats(self) -> dict:
    with self._get_connection() as conn:
        rows = conn.execute("SELECT key, value FROM pipeline_stats").fetchall()
        return dict(rows) if rows else {"total_generated": 0, ...}
```

---

#### H2. Incomplete Type System Enforcement

**Title:** DataType.WINDOW vs DataType.INDEX split exists but not fully validated

**Why it matters:** Prevents nonsensical expressions like `ts_std(close, window=ts_argmax(close, 20))` where window size is dynamic.

**Evidence:**

- `src/alphaforge/discovery/expression/types.py:14-16` — defines `DataType.WINDOW` and `DataType.INDEX` (per `REPORT.md` fix #8)
- Operator signatures use this distinction (lines 40-120)
- BUT: No runtime validator in ExpressionCompiler or TreeGenerator to reject invalid trees
- `test_discovery/test_expression_tree.py` has no test for “invalid window size” rejection

**How to verify:**

```python
# Try creating invalid tree manually
from alphaforge.discovery.expression.nodes import *
root = OperatorNode(
    name="ts_std",
    children=[
        TerminalNode(name="close"),
        OperatorNode(name="ts_argmax", children=[...])  # Invalid: INDEX used as WINDOW
    ]
)
# Does this compile without error? (It shouldn't)
```

**Suggested fix:** Add type validation in `TreeGenerator.generate()` before returning tree

---

#### H3. Event-Driven Engine Execution Realism Gap

**Title:** Event-driven backtest lacks realistic liquidity modeling

**Why it matters:** Claims “production-grade” execution realism but uses fixed market impact formula without order book dynamics.

**Evidence:**

- `src/alphaforge/backtest/event_driven.py:1-180` — basic order lifecycle (PENDING/FILLED)
- `src/alphaforge/backtest/impact.py:50-95` — Almgren-Chriss permanent + temporary impact
- No order book simulation (no bid/ask spread modeling beyond fixed slippage)
- No queue position dynamics
- Partial fills modeled but simplistic (lines 120-135 use coin flip, not realistic volume matching)
- Per `REPORT.md` section 8, this WAS fake (used mock SMA signals), now fixed to execute actual strategy

**Current state vs claims:**

- `README.md` line 458: “Queue position modeling, partial fills, market impact”
- Actual implementation: Simplified versions of these

**Suggested mitigation:** Update README to clarify “simplified execution simulation” or integrate a real simulator (e.g., Zipline, QuantConnect)

---

#### H4. Survivorship Bias Not Addressed (Mock Universe)

**Title:** UniverseRegistry uses mock data, not production index constituents

**Why it matters:** Testing on “S&P 500 current constituents” backtest to 2010 includes survivorship bias (companies that failed are excluded).

**Evidence:**

- `src/alphaforge/data/universe.py:80-120` — `get_constituents()` returns mock tickers
- Line 85 comment: `# TODO: Integrate with real data source (e.g., CRSP, Sharadar)`
- `test_data/test_universe.py:30` — test explicitly uses `use_mock=True`

**How to reproduce:**

```python
from alphaforge.data.universe import UniverseRegistry
constituents = UniverseRegistry.get_constituents("SP500", date="2010-01-01")
# Returns 2025 constituents, not 2010 constituents
```

**Suggested fix:** Integrate with Sharadar/Norgate/CRSP or add prominent warning in README

---

### MEDIUM SEVERITY

#### M1. Stress Test Reproducibility

**Title:** Stress tests use unseeded random number generators

**Why it matters:** Results cannot be reproduced, making debugging impossible.

**Evidence:**

- `src/alphaforge/validation/stress.py:150-180` — synthetic scenarios use `random.Random()` without seed parameter
- `stress.py:165` — `rng = random.Random()` (no seed argument accepted)

**Suggested fix:** Add seed parameter to `StressTester.__init__()` and pass to all RNG instances

---

#### M2. Fitness Cache Collision Risk

**Title:** Discovery fitness cache uses formula string as key

**Why it matters:** Two different trees could theoretically generate identical formulas (e.g., add(x, y) vs add(y, x) if operator is commutative).

**Evidence:**

- `src/alphaforge/discovery/orchestrator.py:245` — `cache_key = tree.formula`
- `tree.formula` is generated from AST traversal (could have ordering issues)

**Suggested fix:** Use `tree.hash` instead of formula for fitness cache (line 328 already does this for compilation cache)

---

#### M3. CORS Hardcoded to Localhost

**Title:** API allows only `http://localhost:3000` origin

**Why it matters:** Cannot deploy frontend to different domain without code change.

**Evidence:**

- `src/alphaforge/api/server.py:33` — `allow_origins=["http://localhost:3000"]`

**Suggested fix:** Use environment variable for allowed origins

---

### LOW SEVERITY

#### L1. Data Cache ID Collision (Mitigated by Weakref)

**Title:** `_get_prepared_data()` uses `id(data)` as cache key

**Why it matters:** Python can reuse object IDs after garbage collection, potentially returning wrong cached DataFrame.

**Evidence:**

- `src/alphaforge/discovery/expression/compiler.py:100` — `cache_key = id(data)`
- Line 105 mitigation: `if ref() is data` — validates weakref before returning

**Actual risk:** Low (weakref check prevents most issues)

**Suggested improvement:** Use `hash(data.index.tobytes() + data.columns.tobytes())` for true content hash

---

#### L2. Test Markers Not Fully Utilized

**Title:** `@pytest.mark.slow` defined but no usage guidance

**Why it matters:** Developers don't know which tests are slow, leading to inefficient test runs.

**Evidence:**

- `pyproject.toml:129` — defines slow marker
- No tests actually marked with `@pytest.mark.slow`

**Suggested fix:** Mark CPCV/PBO/discovery tests as slow, document `pytest -m "not slow"` in README

---

## 5. Doc vs Code Mismatches

| README/REPORT Claim            | Code Reality                                     | Severity          | Location                    |
|--------------------------------|--------------------------------------------------|-------------------|-----------------------------|
| "PBO validation"               | Probability of Loss proxy, not true PBO          | Critical          | validation/pipeline.py:282  |
| "Production-grade platform"    | Research prototype quality                       | Medium            | Throughout                  |
| "Event-driven backtesting"     | Simplified execution simulation                  | High              | backtest/event_driven.py    |
| "Survivorship bias prevention" | Mock universe data only                          | High              | data/universe.py:85         |
| "300+ tests"                   | ~382 tests (7,141 lines)                         | Low (overclaimed) | tests/                      |
| "CPCV 12,870 combinations"     | Actually C(16,8)=12,870 ✓                        | None              | validation/cpcv.py          |
| "Bi-temporal schema"           | Timestamps tracked but NOT enforced in operators | Medium            | data/schema.py, features/   |
| "No synthetic data"            | True for OHLCV; False for FRED/Universe          | Medium            | data/alfred.py, universe.py |
| "Defense-in-depth validation"  | DSR+CPCV+SPA strong; PBO is proxy                | Medium            | validation/                 |

---

## 6. Risk Assessment

### Scientific Validity: 6/10 (Moderate-High Risk)

**Strengths:**

- Lookahead bias fully addressed in signal processing ✓
- DSR and CPCV correctly implemented ✓
- Minimum activity constraints prevent zero-trade gaming ✓
- Point-in-time safety enforced in operators ✓

**Weaknesses:**

- PBO is proxy, not true selection-based measurement ⚠
- Discovery pipeline skips statistical validation entirely ✗
- Survivorship bias not addressed (mock universe) ✗
- Pipeline stats not persisted (DSR underestimation after restart) ⚠

**Verdict:** Suitable for research/paper trading with caveats; NOT production-ready for live capital deployment without addressing discovery validation gap.

---

### Execution Realism: 5/10 (Moderate Risk)

**Strengths:**

- Transaction costs modeled (commission + slippage) ✓
- Market impact formula (Almgren-Chriss) ✓
- Auto-adjusted prices for splits/dividends ✓
- Lag between signal and execution (shift positions) ✓

**Weaknesses:**

- No order book depth modeling ✗
- Partial fills use coin flip, not realistic matching ✗
- No intraday execution (daily bars only) ✗
- No liquidity filters (could generate trades in illiquid stocks) ✗

**Verdict:** Good enough for strategy screening; insufficient for production deployment.

---

### Production Readiness: 3/10 (High Risk)

**Strengths:**

- SQLite persistence for results ✓
- API/frontend architecture in place ✓
- Error handling in background tasks ✓
- Logging infrastructure ✓

**Weaknesses:**

- No job queue (in-memory BackgroundTasks only) ✗
- No multi-worker support (globals would conflict) ✗
- Pipeline stats not persisted ✗
- No authentication/authorization ✗
- No rate limiting ✗
- No deployment infrastructure (docker, k8s, etc.) ✗
- “Deployment” is a placeholder ✗

**Verdict:** Demo/prototype quality. Requires significant infrastructure work for production.

---

## 7. Prioritized Next Steps (Top 10)

| #   | Task                                               | Impact   | Effort | File(s)                                           |
|-----|----------------------------------------------------|----------|--------|---------------------------------------------------|
| 1   | Wire discovery pipeline to ValidationPipeline      | Critical | Medium | api/server.py:520-576, discovery/orchestrator.py  |
| 2   | Persist pipeline_stats to database                 | Critical | Low    | api/storage.py, api/server.py:50-56               |
| 3   | Rename "PBO" to "Probability of Loss" in API/UI    | Critical | Low    | validation/pipeline.py, frontend/                 |
| 4   | Implement true PBO for multi-strategy discovery    | High     | High   | validation/pipeline.py, discovery/orchestrator.py |
| 5   | Add survivorship bias warning to README            | High     | Low    | README.md                                         |
| 6   | Integrate real universe data (Sharadar/CRSP)       | High     | High   | data/universe.py                                  |
| 7   | Add type validation to TreeGenerator               | High     | Medium | discovery/expression/tree.py                      |
| 8   | Make stress tests reproducible (add seed)          | Medium   | Low    | validation/stress.py                              |
| 9   | Use tree.hash instead of formula for fitness cache | Medium   | Low    | discovery/orchestrator.py:245                     |
| 10  | Add deployment warning/disclaimer to README        | Medium   | Low    | README.md                                         |

---

## 8. Open Questions (Require Further Investigation)

1. Corporate actions beyond splits/dividends: Does yfinance handle spin-offs, mergers, ticker changes correctly? Need to validate against known events (e.g., Google → Alphabet).
2. Multi-worker safety: What happens if two FastAPI workers access pipeline_stats dict simultaneously? Need to test with `uvicorn --workers=4`.
3. CPCV computational limits: At what strategy count does CPCV become prohibitive? README claims `max_combinations=1000 for speed` but doesn't document performance characteristics.
4. Discovery ensemble weights: How are ensemble_weights calculated (line 199 of orchestrator.py)? Code calls `_create_ensemble()` but implementation needs review.
5. Genetic algorithm vs NSGA-III convergence: Factory uses single-objective "sharpe", Discovery uses multi-objective. Are they comparable? Should Factory also use NSGA-III?
6. MLflow integration: tracking/mlflow.py exists but not called in pipeline. Is experiment tracking functional?
7. Walk-forward analysis: Module exists (`validation/walk_forward.py`) but not integrated. Was this deprioritized?
8. Data quality validation: `data/quality.py` implements checks but never called in loader. Should this be automatic?
9. Numba compilation overhead: Compiler uses `@numba.jit(cache=True)`. Does cache warming happen? First-run performance impact?
10. True signal-to-position logic: Discovery uses threshold at ±0.5σ. Should this be configurable? What about continuous position sizing?

---

## Summary & Recommendation

AlphaForge is a well-architected research platform with strong fundamentals (anti-lookahead, CPCV, DSR) but critical gaps prevent "production-grade" deployment:

1. Discovery pipeline lacks validation → Systematic overfit risk
2. PBO is mislabeled → Users may misunderstand robustness claims
3. Execution realism is simplified → Paper trading only
4. Survivorship bias unaddressed → Backtest results optimistic

Recommended path forward:

- Short-term (1-2 weeks): Fix items #1-3, #8-10 above (validation wiring, stats persistence, naming)
- Medium-term (1-2 months): Items #4-7 (true PBO, universe data, type validation)
- Long-term (3-6 months): Production infrastructure (job queue, auth, deployment, liquidity modeling)

Current best use: Academic research, strategy prototyping, educational tool  
NOT ready for: Live trading, capital deployment, commercial SaaS offering

