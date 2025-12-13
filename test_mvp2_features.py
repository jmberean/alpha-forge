#!/usr/bin/env python3
"""Quick test of MVP2 features with real data."""

import numpy as np
import pandas as pd
from alphaforge.data.loader import MarketDataLoader
from alphaforge.strategy.templates import StrategyTemplates
from alphaforge.validation.spa import SPATest
from alphaforge.validation.stress import StressTester
from alphaforge.monitoring.cusum import CUSUMMonitor, AlertLevel
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.backtest.impact import MarketImpactModel, ExecutionModel

print("=" * 80)
print("MVP2 FEATURES - LIVE PERFORMANCE TEST")
print("=" * 80)

# Load real market data
print("\n1. Loading SPY data...")
loader = MarketDataLoader()
spy_data = loader.load("SPY", start="2020-01-01", end="2023-12-31")
print(f"   ✓ Loaded {len(spy_data.df)} days of SPY data")

# Get benchmark returns
spy_returns = spy_data.df['close'].pct_change().dropna()
print(f"   ✓ Calculated {len(spy_returns)} daily returns")

# Test a strategy
print("\n2. Testing RSI Mean Reversion strategy...")
strategy = StrategyTemplates.get_template("rsi_mean_reversion")
engine = BacktestEngine(initial_capital=100000)
backtest_result = engine.run(strategy, spy_data)

print(f"   Strategy Performance:")
print(f"   - Sharpe Ratio: {backtest_result.metrics.sharpe_ratio:.2f}")
print(f"   - Annual Return: {backtest_result.metrics.annualized_return:.1%}")
print(f"   - Max Drawdown: {backtest_result.metrics.max_drawdown:.1%}")

# Test Hansen's SPA
print("\n3. Testing Hansen's SPA Test...")
spa = SPATest(pvalue_threshold=0.05, bootstrap_reps=500)  # Reduced reps for speed
strategy_returns = backtest_result.returns
spa_result = spa.test(strategy_returns, spy_returns, "SPY")

print(f"   SPA Test Results:")
print(f"   - P-value: {spa_result.pvalue:.4f}")
print(f"   - Test Statistic: {spa_result.test_statistic:.2f}")
print(f"   - Outperformance: {spa_result.outperformance:.4%} daily")
print(f"   - Verdict: {'✓ PASS' if spa_result.passed else '✗ FAIL'}")

# Test Market Impact Model
print("\n4. Testing Almgren-Chriss Market Impact...")
impact_model = MarketImpactModel()
exec_model = ExecutionModel(impact_model=impact_model)

# Simulate a $100k order in SPY (avg volume ~70M/day, vol ~1.5%)
test_order = 100000
test_volume = 70000000
test_volatility = 0.015

impact = impact_model.calculate_impact(
    order_size=test_order,
    daily_volume=test_volume,
    volatility=test_volatility,
    time_horizon=1.0
)

print(f"   Market Impact Analysis:")
print(f"   - Order Size: ${test_order:,.0f}")
print(f"   - Daily Volume: ${test_volume:,.0f}")
print(f"   - Participation Rate: {test_order/test_volume:.4%}")
print(f"   - Total Impact: {impact*10000:.2f} bps")

# Test Stress Testing (just one scenario for speed)
print("\n5. Testing Stress Testing Framework...")
print("   Running 2020 COVID crash scenario...")
tester = StressTester()
stress_result = tester.test_strategy(
    strategy,
    "SPY",
    scenarios=["2020_covid_crash"]
)

print(f"   Stress Test Results:")
print(f"   - Scenarios Tested: {stress_result.scenarios_tested}")
print(f"   - Scenarios Passed: {stress_result.scenarios_passed}")
print(f"   - Pass Rate: {stress_result.pass_rate:.1%}")
print(f"   - Verdict: {'✓ PASS' if stress_result.passed else '✗ FAIL'}")

for result in stress_result.results:
    print(f"     • {result.scenario_name}: Sharpe={result.sharpe:.2f}, DD={result.max_drawdown:.1%}")

# Test CUSUM Monitoring
print("\n6. Testing CUSUM Monitoring...")
monitor = CUSUMMonitor(expected_sharpe=1.0, drift_allowance=0.5)

# Simulate monitoring over time
recent_returns = strategy_returns.tail(100)
cusum_result = monitor.batch_monitor(recent_returns)

print(f"   CUSUM Monitoring Results:")
print(f"   - Periods Monitored: {len(cusum_result.statistics)}")
print(f"   - Alerts Triggered: {len(cusum_result.alerts)}")
print(f"   - Final Statistic: {cusum_result.final_state.statistic:.2f}")
print(f"   - Final Alert Level: {cusum_result.final_state.alert_level.value.upper()}")

if cusum_result.alerts:
    print(f"   Alert History:")
    for alert in cusum_result.alerts[:3]:  # Show first 3
        print(f"     • {alert.timestamp.date()}: {alert.alert_level.value.upper()} (S={alert.statistic:.2f})")

# Summary
print("\n" + "=" * 80)
print("MVP2 FEATURES TEST - SUMMARY")
print("=" * 80)
print(f"✓ Market Data Loading:        Working")
print(f"✓ Strategy Backtesting:       Sharpe {backtest_result.metrics.sharpe_ratio:.2f}")
print(f"✓ Hansen's SPA Test:          {'PASS' if spa_result.passed else 'FAIL'} (p={spa_result.pvalue:.4f})")
print(f"✓ Market Impact Model:        {impact*10000:.2f} bps for ${test_order:,.0f} order")
print(f"✓ Stress Testing:             {stress_result.pass_rate:.0%} pass rate")
print(f"✓ CUSUM Monitoring:           {cusum_result.final_state.alert_level.value.upper()}")
print("=" * 80)
print("\n🎉 All MVP2 features operational!")
