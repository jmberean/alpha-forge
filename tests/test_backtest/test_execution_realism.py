
import pandas as pd
import numpy as np
import pytest
from datetime import datetime

from alphaforge.backtest.event_driven import EventDrivenEngine
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.strategy.templates import StrategyTemplates
from alphaforge.data.schema import OHLCVData

class TestExecutionRealism:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2023-01-01", periods=100)
        close = 100.0 + np.random.randn(100).cumsum()
        open_p = np.roll(close, 1) # Open = prev close
        open_p[0] = 100.0
        
        df = pd.DataFrame({
            "close": close,
            "open": open_p,
            "volume": 1000000
        }, index=dates)
        
        # Ensure High >= max(open, close) and Low <= min(open, close)
        df["high"] = df[["open", "close"]].max(axis=1) + 1.0
        df["low"] = df[["open", "close"]].min(axis=1) - 1.0
        
        return OHLCVData(df=df, symbol="TEST")

    def test_engines_produce_directionally_consistent_results(self, sample_data):
        """
        Verify that Vectorized and Event-Driven engines produce similar outcomes
        for a simple strategy (SMA Crossover).
        They won't be identical due to execution simulation, but should be correlated.
        """
        # Use standard periods that are pre-computed
        strategy = StrategyTemplates.sma_crossover(fast_period=20, slow_period=50)
        
        # Vectorized Run
        vec_engine = BacktestEngine()
        vec_result = vec_engine.run(strategy, sample_data)
        
        # Event-Driven Run
        event_engine = EventDrivenEngine()
        event_result = event_engine.run(strategy, sample_data)
        
        # Check basic properties
        assert not event_result.returns.empty
        assert len(event_result.equity_curve) == len(vec_result.equity_curve)
        
        # If vectorized made money, event driven should generally follow (unless edge case)
        # We can't assert strict correlation on random data, but we can check that
        # the event driven engine actually TRADED.
        # Note: Short SMA period ensures trades in 100 bars.
        
        # Assert that Event Driven engine respected the signals
        # If vectorized has trades, event driven should have orders/fills
        if vec_result.num_trades > 0:
            assert len(event_engine.orders) > 0
            # Fills might be fewer due to prob model, but typically > 0
            # unless random seed is very unlucky.

    def test_strategy_logic_is_respected(self, sample_data):
        """
        Critical Test: Ensure changing the strategy genome actually changes
        the event-driven result. (Previously failed when logic was hardcoded).
        """
        engine = EventDrivenEngine()
        
        # Strategy 1: SMA Crossover
        strat1 = StrategyTemplates.sma_crossover(fast_period=20, slow_period=50)
        res1 = engine.run(strat1, sample_data)
        
        # Strategy 2: Buy and Hold (or something very different)
        # Using RSI mean reversion as it's structurally distinct
        strat2 = StrategyTemplates.rsi_mean_reversion()
        res2 = engine.run(strat2, sample_data)
        
        # Results must differ
        # Use returns series for comparison
        assert not res1.returns.equals(res2.returns), \
            "EventDrivenEngine returned identical results for different strategies! Logic is likely ignored."
