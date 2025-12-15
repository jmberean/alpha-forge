# AlphaForge Review: Strategic Analysis

## 1. Goal Fit (Verdict)
**Verdict: Promising Core, Dangerous Gaps.**
AlphaForge aspires to be a "production-grade" platform but is currently a **high-quality research prototype** with significant gaps in execution realism and production readiness.

*   **Scientific Validity:** **Strong (with caveats).** DSR and CPCV are implemented and tested. Selection-based PBO exists (matrix method), but the main pipeline currently reports a single-strategy robustness proxy (“probability of loss”) unless population-level selection is explicitly analyzed.
*   **Execution Realism:** **Weak.** The "Event-Driven" engine is a rudimentary simulation. It lacks real order book modeling, realistic liquidity constraints, and fails to handle corporate actions or survivorship bias.
*   **Production Readiness:** **Aspirational.** "Deployment" is a placeholder. There is no persistence layer (all in-memory), no job queue, and no safety rails for live trading (kill switches, position limits).
*   **Workflow:** The end-to-end workflow (Discover -> Validate) functions well for research but stops abruptly before "Deploy."

## 2. Strengths (Evidence-Based)
*   **Anti-Leakage Operators:** `src/alphaforge/discovery/expression/compiler.py` and `src/alphaforge/features/technical.py` strictly use trailing windows (`rolling(..., center=False)`), preventing future data leakage in feature construction. (Note: The underlying data object `OHLCVData` still has some point-in-time design gaps).
*   **Rigorous Statistical Validation:** Deflated Sharpe Ratio (`validation/dsr.py`) and CPCV (`validation/cpcv.py`) are faithful building blocks; PBO’s matrix method exists in `validation/pbo.py`, while the pipeline’s default single-strategy output is a probability-of-loss proxy (by design, to avoid mislabeling).
*   **Data Integrity:** The data loader (`data/loader.py`) cleanly separates raw data ingestion from feature engineering, minimizing the risk of "snooping."
*   **Strongly Typed GP:** The genetic programming engine (`discovery/expression/`) uses a robust type system that prevents syntax errors during evolution.

## 3. Data Sources (Verified)
*   **Active:** **Yahoo Finance** (via `yfinance`) is the sole provider of real-world OHLCV data. It is cached locally in Parquet format to support reproducible research.
*   **Mocked (Planned):**
    *   **FRED (Federal Reserve Economic Data):** The `ALFREDClient` is implemented but currently uses deterministic mock data for economic indicators (GDP, CPI, etc.).
    *   **Index Constituents:** The `UniverseRegistry` is designed to prevent survivorship bias by tracking historical index membership (e.g., deleted stocks), but currently relies on mock data. Production use would require a subscription to CRSP, Sharadar, or Norgate.

## 4. Gaps & Risks (Prioritized)

### **Critical Severity**
1.  **The "Zero-Trade" Loophole (Gaming the Optimizer):**
    *   **What:** Strategies with 0 trades get perfect scores for Drawdown and Turnover, allowing them to dominate the Pareto front.
    *   **Why:** This wastes computational resources and produces useless "do-nothing" strategies.
    *   **Fix:** Implement a "Minimum Activity Constraint" in the fitness function (e.g., Sharpe = -999 if trades < 10).
2.  **Lookahead Bias in Discovery (Global Normalization):**
    *   **What:** `_signals_to_positions` in `discovery/orchestrator.py` standardizes signals using the *entire* series mean and std (`(signal - mean) / std`).
    *   **Why:** This leaks future information (e.g., future volatility regimes) into past decisions, invalidating all discovery results.
    *   **Fix:** Replace global normalization with a rolling or expanding window z-score.
3.  **Misimplemented PBO/CPCV:**
    *   **What:** The current "PBO" implementation calculates the fraction of test splits with negative Sharpe ratios.
    *   **Why:** True PBO (Bailey et al.) measures the probability that the *best In-Sample strategy* underperforms Out-of-Sample. The current metric is just "Probability of Loss," not "Probability of Overfitting," and fails to account for selection bias.
    *   **Fix:** Re-implement PBO to compare In-Sample vs. Out-of-Sample ranks across multiple strategy variations.
4.  **No Persistence Layer:**
    *   **What:** `src/alphaforge/api/server.py` stores all results in global Python dictionaries (`validation_results = {}`).
    *   **Why:** If the server restarts or scales to multiple workers, all validation history and discovered strategies are lost.
    *   **Fix:** Integrate a lightweight database (SQLite/PostgreSQL) or file-based persistence.

### **High Severity**
5.  **Redundant Evolution Engines:**
    *   **What:** `discovery` uses NSGA-III (custom), while `factory` uses DEAP (genetic algorithm).
    *   **Why:** Doubles maintenance, inconsistent results, and missed opportunity for unified "Architecture + Parameter" optimization.
    *   **Fix:** Consolidate into a single `AlphaEngine`.
6.  **Fake Execution Engine:**
    *   **What:** `backtest/event_driven.py` ignores the passed `StrategyGenome` and instead generates signals using hardcoded mock SMA logic (`if sma_fast > sma_slow`).
    *   **Why:** The "Event-Driven Backtest" provides no information about the actual strategy being validated. It is a placeholder that actively misleads the user.
    *   **Fix:** Wire the `SignalGenerator` into the event-driven loop so it executes the actual strategy logic.
7.  **Data Realism (Splits/Dividends):**
    *   **What:** `MarketDataLoader` uses `auto_adjust=False` and backtests calculate returns on raw `close` prices.
    *   **Why:** Stock splits and dividends will appear as massive price shocks (e.g., a 2:1 split looks like a 50% crash), completely destroying backtest validity for longer horizons.
    *   **Fix:** Enable `auto_adjust=True` or explicitly handle adjusted close columns in the backtest engine.

### **Medium Severity**
8.  **Type System Ambiguity:**
    *   **What:** `DataType.INTEGER` treats Window Size and Time Index identically.
    *   **Why:** Allows nonsensical formulas like `std(close, window=days_since_high)`.
    *   **Fix:** Split into `DataType.WINDOW` (static) and `DataType.INDEX` (dynamic).
9.  **Reproducibility Gaps:**
    *   **What:** Discovery runs default to `seed=None`, and stress tests use unseeded randomness.
    *   **Why:** Results cannot be replicated, making debugging and scientific verification impossible.
    *   **Fix:** Enforce explicit random seeds for all stochastic components (Discovery, Factory, Stress Tests).

## 5. Better Approaches (Top Recommendations)
1.  **Unified "Alpha Engine":**
    *   **Proposal:** Merge `Discovery` and `Factory`. Treat "Templates" as pre-seeded `ExpressionTrees` in the genetic population.
    *   **Tradeoff:** Higher initial refactoring cost vs. long-term simplicity and power.
    *   **Recommendation:** **Do this first.** It solves the architectural debt.
2.  **True PBO Implementation:**
    *   **Proposal:** Implement the "Matrix Method" from Bailey et al. Calculate In-Sample and Out-of-Sample ranks for *all* strategy variations across *all* splits to measure the probability of selection bias.
    *   **Tradeoff:** Computationally expensive (requires storing full history for population).
    *   **Recommendation:** Essential for the "Defense-in-Depth" value proposition.
3.  **Database-Backed Job Queue:**
    *   **Proposal:** Replace `BackgroundTasks` + in-memory dicts with Celery/Redis or just a robust SQL-backed job table.
    *   **Tradeoff:** Infrastructure complexity vs. reliability.
    *   **Recommendation:** Essential for "production" claims.
4.  **Strict "Activity" Gate:**
    *   **Proposal:** Hard-fail any strategy with < N trades or < X% volatility in the fitness function.
    *   **Tradeoff:** Might kill niche "rare event" strategies.
    *   **Recommendation:** Necessary to stop the optimizer from gaming the system.

## 6. Roadmap (Next Actions)
1.  **Fix Logic:** Add "Minimum Activity Constraint" to `discovery/orchestrator.py`.
2.  **Fix Types:** Split `DataType.INTEGER` into `WINDOW` and `INDEX`.
3.  **Refactor:** Merge `Factory` and `Discovery` into `AlphaEngine`.
4.  **Persistence:** Replace in-memory storage with SQLite/SQLModel.
5.  **Realism:** Upgrade `EventDrivenEngine` or integrate a third-party execution simulator.
6.  **Hardening:** Add "kill switches" and position limits to the `BacktestEngine`.

## 7. Open Questions / Not Confirmed
*   **Survivorship Bias:** I did not see explicit handling of delisted assets in the `MarketDataLoader`. This needs verification.
*   **Corporate Actions:** It is unclear if splits/dividends are handled correctly beyond what `yfinance` provides.
*   **Data Snooping:** While *feature* leakage is handled, I need to verify if *parameter* tuning (in `Factory`) reuses the same test set too many times without correcting DSR.

---

## 8. Implementation Update (2025-12-15)

This section summarizes what has been implemented since this report was written (including changes made by another LLM + follow-up fixes).

### Completed / Partially Completed Items From This Report

1. **Fix Logic: Minimum Activity Constraint (Roadmap #1, Critical #1)**
   - Implemented `min_trades` and `min_volatility` in `DiscoveryConfig` and enforced a hard penalty in `DiscoveryOrchestrator._compute_all_fitness` when strategies have insufficient activity.
   - Files: `src/alphaforge/discovery/orchestrator.py`, `tests/test_discovery/test_orchestrator_logic.py`.

2. **Fix Logic: Lookahead Bias in Discovery Normalization (Critical #2)**
   - Replaced global normalization in `_signals_to_positions` with an expanding-window z-score (point-in-time safe).
   - Added regression tests asserting prefix invariance (positions for `t<=T` unchanged when future data is appended).
   - Files: `src/alphaforge/discovery/orchestrator.py`, `tests/test_discovery/test_orchestrator_logic.py`.

3. **Misimplemented PBO/CPCV (Critical #3) — Partially Addressed**
   - The CPCV-based “negative Sharpe fraction” metric was renamed to **Probability of Loss** to avoid mislabeling it as true PBO.
   - `ValidationPipeline` now logs this explicitly as a “single-strategy PBO proxy”.
   - True selection-based PBO still exists as a matrix method in `ProbabilityOfOverfitting.calculate`, but it is not yet wired into the main pipeline for “population-level” evaluation.
   - Files: `src/alphaforge/validation/pbo.py`, `src/alphaforge/validation/pipeline.py`.

4. **No Persistence Layer (Critical #4)**
   - Added SQLite persistence via `Storage` and updated the API to read/write persisted results instead of relying on in-memory globals.
   - Validation results are persisted; factory + discovery runs are now also persisted.
   - Files: `src/alphaforge/api/storage.py`, `src/alphaforge/api/server.py`, `tests/test_api/test_persistence.py`.

5. **Redundant Evolution Engines (High #5) — Partially Addressed**
   - Introduced a minimal `Evolvable` protocol plus `ExpressionGenome` and `TemplateGenome` wrappers to support using a single NSGA-III optimizer for both expression discovery and template parameter evolution.
   - Factory genetic evolution was migrated to NSGA-III (single-objective “sharpe”) using `TemplateGenome`. Some `TemplateGenome` mutation logic remains stubby/limited.
   - Files: `src/alphaforge/evolution/protocol.py`, `src/alphaforge/evolution/genomes.py`, `src/alphaforge/factory/orchestrator.py`, `src/alphaforge/discovery/evolution/nsga3.py`, `src/alphaforge/discovery/evolution/population.py`.

6. **Fake Execution Engine (High #6)**
   - `EventDrivenEngine` now executes the passed `StrategyGenome` (uses `SignalGenerator` + computed indicators) instead of hardcoded SMA mock logic, and returns a fully populated `BacktestResult`.
   - Files: `src/alphaforge/backtest/event_driven.py`, `tests/test_backtest/test_event_driven.py`, `tests/test_backtest/test_execution_realism.py`.

7. **Data Realism: Splits/Dividends (High #7)**
   - `MarketDataLoader` now fetches adjusted OHLC via `yfinance.history(auto_adjust=True)` to reduce split/dividend artifacts.
   - File: `src/alphaforge/data/loader.py`.

8. **Type System Ambiguity (Medium #8)**
   - Split `DataType.INTEGER` into `DataType.WINDOW` (static constant) and `DataType.INDEX` (dynamic integer series).
   - Updated operator signatures and constant node defaults accordingly.
   - Files: `src/alphaforge/discovery/expression/types.py`, `src/alphaforge/discovery/expression/nodes.py`, plus associated discovery tests.

### Additional Goal-Critical Fixes (Not Explicitly Listed Above)

- **Discovery “cross-sectional ops” made point-in-time safe for single-series runs**
  - Implemented trailing-window approximations for `rank/scale/zscore` to avoid full-sample leakage in a single time series context.
  - Added a PIT safety regression test for `rank`.
  - Files: `src/alphaforge/discovery/expression/compiler.py`, `tests/test_discovery/test_expression_tree.py`.

- **Expression compiler cache correctness**
  - Fixed a potential correctness bug where `_get_prepared_data` cached by `id(data)` could return the wrong prepared DataFrame if Python reused object IDs.
  - Files: `src/alphaforge/discovery/expression/compiler.py`.

- **Quality reporting semantics**
  - Updated `QualityReport.has_errors` to treat warnings as “errors” for strict-mode handling.
  - File: `src/alphaforge/data/quality.py`.

- **Runtime path consistency**
  - Updated startup scripts to avoid `cd src` so data/db/log paths remain consistent.
  - Files: `start.sh`, `start-dev.sh`, `.gitignore`.

### Frontend UX Improvements

- Validation view now renders the backend-provided equity curve and metrics rather than showing synthetic chart data.
- Files: `frontend/components/BacktestChart.tsx`, `frontend/components/ValidationRunner.tsx`, `frontend/lib/api.ts`.

### Current Test Status

- `pytest` passes (382 tests).

### Documentation Notes

- Added `AGENTS.md` (contributor guidelines).
- Consolidated the separate logic review into this file; see Sections 1–7 and the implementation update above.
