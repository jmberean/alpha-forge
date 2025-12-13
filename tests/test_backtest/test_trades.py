"""Tests for trade analysis."""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from alphaforge.backtest.trades import (
    Trade,
    TradeAnalyzer,
    TradeStats,
    analyze_trades,
)


@pytest.fixture
def sample_trades():
    """Generate sample trades for testing."""
    base_date = datetime(2020, 1, 1)

    trades = [
        # Winning trades
        Trade(
            entry_date=base_date,
            entry_price=100.0,
            entry_signal="long",
            exit_date=base_date + timedelta(days=5),
            exit_price=110.0,
            exit_signal="target",
            pnl=10.0 * 100,  # 10% gain
            pnl_pct=0.10,
            return_=0.10,
            size=100.0,
            duration=5,
        ),
        Trade(
            entry_date=base_date + timedelta(days=10),
            entry_price=110.0,
            entry_signal="long",
            exit_date=base_date + timedelta(days=15),
            exit_price=120.0,
            exit_signal="signal",
            pnl=10.0 * 100,
            pnl_pct=0.091,
            return_=0.091,
            size=100.0,
            duration=5,
        ),
        # Losing trades
        Trade(
            entry_date=base_date + timedelta(days=20),
            entry_price=120.0,
            entry_signal="long",
            exit_date=base_date + timedelta(days=22),
            exit_price=115.0,
            exit_signal="stop",
            pnl=-5.0 * 100,
            pnl_pct=-0.042,
            return_=-0.042,
            size=100.0,
            duration=2,
        ),
        Trade(
            entry_date=base_date + timedelta(days=25),
            entry_price=115.0,
            entry_signal="long",
            exit_date=base_date + timedelta(days=27),
            exit_price=112.0,
            exit_signal="stop",
            pnl=-3.0 * 100,
            pnl_pct=-0.026,
            return_=-0.026,
            size=100.0,
            duration=2,
        ),
        # Breakeven trade
        Trade(
            entry_date=base_date + timedelta(days=30),
            entry_price=112.0,
            entry_signal="long",
            exit_date=base_date + timedelta(days=31),
            exit_price=112.0,
            exit_signal="time",
            pnl=0.0,
            pnl_pct=0.0,
            return_=0.0,
            size=100.0,
            duration=1,
        ),
    ]

    return trades


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test creating a Trade object."""
        trade = Trade(
            entry_date=datetime(2020, 1, 1),
            entry_price=100.0,
            entry_signal="long",
            exit_date=datetime(2020, 1, 5),
            exit_price=105.0,
            exit_signal="target",
            pnl=5.0,
            pnl_pct=0.05,
            return_=0.05,
            size=1.0,
            duration=4,
        )

        assert trade.entry_price == 100.0
        assert trade.exit_price == 105.0
        assert trade.pnl == 5.0
        assert trade.duration == 4

    def test_is_winner(self):
        """Test is_winner method."""
        winner = Trade(
            entry_date=datetime(2020, 1, 1),
            entry_price=100.0,
            entry_signal="long",
            exit_date=datetime(2020, 1, 5),
            exit_price=105.0,
            exit_signal="target",
            pnl=5.0,
            pnl_pct=0.05,
            return_=0.05,
            size=1.0,
            duration=4,
        )

        loser = Trade(
            entry_date=datetime(2020, 1, 1),
            entry_price=100.0,
            entry_signal="long",
            exit_date=datetime(2020, 1, 5),
            exit_price=95.0,
            exit_signal="stop",
            pnl=-5.0,
            pnl_pct=-0.05,
            return_=-0.05,
            size=1.0,
            duration=4,
        )

        assert winner.is_winner() is True
        assert loser.is_winner() is False

    def test_to_dict(self):
        """Test to_dict method."""
        trade = Trade(
            entry_date=datetime(2020, 1, 1),
            entry_price=100.0,
            entry_signal="long",
            exit_date=datetime(2020, 1, 5),
            exit_price=105.0,
            exit_signal="target",
            pnl=5.0,
            pnl_pct=0.05,
            return_=0.05,
            size=1.0,
            duration=4,
        )

        trade_dict = trade.to_dict()

        assert trade_dict["entry_price"] == 100.0
        assert trade_dict["exit_price"] == 105.0
        assert trade_dict["pnl"] == 5.0
        assert "entry_date" in trade_dict
        assert "exit_date" in trade_dict


class TestTradeAnalyzer:
    """Test TradeAnalyzer class."""

    def test_analyze_basic_stats(self, sample_trades):
        """Test basic statistics calculation."""
        stats = TradeAnalyzer.analyze(sample_trades)

        assert stats.total_trades == 5
        assert stats.winning_trades == 2
        assert stats.losing_trades == 2
        assert stats.breakeven_trades == 1

    def test_win_rate(self, sample_trades):
        """Test win rate calculation."""
        stats = TradeAnalyzer.analyze(sample_trades)

        # 2 winners out of 5 trades = 40%
        assert stats.win_rate == 0.4
        assert stats.loss_rate == 0.4

    def test_profit_metrics(self, sample_trades):
        """Test profit calculations."""
        stats = TradeAnalyzer.analyze(sample_trades)

        # Total PnL: 1000 + 1000 - 500 - 300 + 0 = 1200
        assert stats.total_pnl == 1200.0

        # Average PnL: 1200 / 5 = 240
        assert stats.avg_pnl == 240.0

        # Average win: (1000 + 1000) / 2 = 1000
        assert stats.avg_win == 1000.0

        # Average loss: (-500 + -300) / 2 = -400
        assert stats.avg_loss == -400.0

    def test_profit_factor(self, sample_trades):
        """Test profit factor calculation."""
        stats = TradeAnalyzer.analyze(sample_trades)

        # Gross profit: 2000
        # Gross loss: 800
        # Profit factor: 2000 / 800 = 2.5
        assert stats.profit_factor == 2.5

    def test_expectancy(self, sample_trades):
        """Test expectancy calculation."""
        stats = TradeAnalyzer.analyze(sample_trades)

        # Expectancy = average PnL = 240
        assert stats.expectancy == 240.0

    def test_duration_metrics(self, sample_trades):
        """Test duration calculations."""
        stats = TradeAnalyzer.analyze(sample_trades)

        # Average duration: (5 + 5 + 2 + 2 + 1) / 5 = 3
        assert stats.avg_duration == 3.0

        # Average win duration: (5 + 5) / 2 = 5
        assert stats.avg_win_duration == 5.0

        # Average loss duration: (2 + 2) / 2 = 2
        assert stats.avg_loss_duration == 2.0

    def test_win_loss_ratio(self, sample_trades):
        """Test win/loss ratio."""
        stats = TradeAnalyzer.analyze(sample_trades)

        # Win/loss ratio: 1000 / 400 = 2.5
        assert stats.avg_win_loss_ratio == 2.5

    def test_largest_win_loss(self, sample_trades):
        """Test largest win and loss tracking."""
        stats = TradeAnalyzer.analyze(sample_trades)

        assert stats.largest_win == 1000.0
        assert stats.largest_loss == -500.0

    def test_consecutive_wins(self):
        """Test max consecutive wins calculation."""
        base_date = datetime(2020, 1, 1)
        trades = [
            Trade(
                entry_date=base_date,
                entry_price=100.0,
                entry_signal="long",
                exit_date=base_date + timedelta(days=1),
                exit_price=105.0,
                exit_signal="target",
                pnl=5.0,
                pnl_pct=0.05,
                return_=0.05,
                size=1.0,
                duration=1,
            ),
            Trade(
                entry_date=base_date + timedelta(days=2),
                entry_price=105.0,
                entry_signal="long",
                exit_date=base_date + timedelta(days=3),
                exit_price=110.0,
                exit_signal="target",
                pnl=5.0,
                pnl_pct=0.048,
                return_=0.048,
                size=1.0,
                duration=1,
            ),
            Trade(
                entry_date=base_date + timedelta(days=4),
                entry_price=110.0,
                entry_signal="long",
                exit_date=base_date + timedelta(days=5),
                exit_price=115.0,
                exit_signal="target",
                pnl=5.0,
                pnl_pct=0.045,
                return_=0.045,
                size=1.0,
                duration=1,
            ),
            Trade(
                entry_date=base_date + timedelta(days=6),
                entry_price=115.0,
                entry_signal="long",
                exit_date=base_date + timedelta(days=7),
                exit_price=110.0,
                exit_signal="stop",
                pnl=-5.0,
                pnl_pct=-0.043,
                return_=-0.043,
                size=1.0,
                duration=1,
            ),
        ]

        stats = TradeAnalyzer.analyze(trades)
        assert stats.max_consecutive_wins == 3
        assert stats.max_consecutive_losses == 1

    def test_empty_trades(self):
        """Test analysis with no trades."""
        stats = TradeAnalyzer.analyze([])

        assert stats.total_trades == 0
        assert stats.winning_trades == 0
        assert stats.win_rate == 0.0
        assert stats.total_pnl == 0.0
        assert stats.profit_factor == 0.0

    def test_only_winners(self):
        """Test analysis with only winning trades."""
        base_date = datetime(2020, 1, 1)
        trades = [
            Trade(
                entry_date=base_date,
                entry_price=100.0,
                entry_signal="long",
                exit_date=base_date + timedelta(days=1),
                exit_price=105.0,
                exit_signal="target",
                pnl=5.0,
                pnl_pct=0.05,
                return_=0.05,
                size=1.0,
                duration=1,
            ),
            Trade(
                entry_date=base_date + timedelta(days=2),
                entry_price=105.0,
                entry_signal="long",
                exit_date=base_date + timedelta(days=3),
                exit_price=110.0,
                exit_signal="target",
                pnl=5.0,
                pnl_pct=0.048,
                return_=0.048,
                size=1.0,
                duration=1,
            ),
        ]

        stats = TradeAnalyzer.analyze(trades)
        assert stats.win_rate == 1.0
        assert stats.profit_factor == float("inf")  # No losses

    def test_only_losers(self):
        """Test analysis with only losing trades."""
        base_date = datetime(2020, 1, 1)
        trades = [
            Trade(
                entry_date=base_date,
                entry_price=100.0,
                entry_signal="long",
                exit_date=base_date + timedelta(days=1),
                exit_price=95.0,
                exit_signal="stop",
                pnl=-5.0,
                pnl_pct=-0.05,
                return_=-0.05,
                size=1.0,
                duration=1,
            ),
        ]

        stats = TradeAnalyzer.analyze(trades)
        assert stats.win_rate == 0.0
        assert stats.profit_factor == 0.0


class TestExtractTrades:
    """Test extracting trades from position series."""

    def test_simple_long_trade(self):
        """Test extracting a simple long trade."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        positions = pd.Series([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], index=dates)
        prices = pd.Series([100, 100, 100, 102, 105, 105, 105, 105, 105, 105], index=dates)

        trades = TradeAnalyzer.extract_trades_from_positions(positions, prices)

        assert len(trades) == 1
        assert trades[0].entry_signal == "long"
        assert trades[0].entry_price == 100.0
        assert trades[0].exit_price == 105.0
        assert trades[0].pnl > 0  # Profitable trade

    def test_multiple_trades(self):
        """Test extracting multiple trades."""
        dates = pd.date_range("2020-01-01", periods=15, freq="D")
        # Two separate trades
        positions = pd.Series(
            [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 0, 0, 0],
            index=dates
        )
        prices = pd.Series(range(100, 115), index=dates)

        trades = TradeAnalyzer.extract_trades_from_positions(positions, prices)

        assert len(trades) == 3  # Three separate trades
        assert trades[0].entry_signal == "long"
        assert trades[1].entry_signal == "short"
        assert trades[2].entry_signal == "long"

    def test_empty_positions(self):
        """Test with no positions taken."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        positions = pd.Series([0] * 10, index=dates)
        prices = pd.Series(range(100, 110), index=dates)

        trades = TradeAnalyzer.extract_trades_from_positions(positions, prices)

        assert len(trades) == 0


class TestConvenienceFunction:
    """Test convenience function."""

    def test_analyze_trades_function(self, sample_trades):
        """Test analyze_trades convenience function."""
        stats = analyze_trades(sample_trades)

        assert isinstance(stats, TradeStats)
        assert stats.total_trades == 5
        assert stats.win_rate == 0.4


class TestTradesDataFrame:
    """Test conversion to DataFrame."""

    def test_trades_to_dataframe(self, sample_trades):
        """Test converting trades to DataFrame."""
        df = TradeAnalyzer.trades_to_dataframe(sample_trades)

        assert len(df) == 5
        assert "entry_price" in df.columns
        assert "exit_price" in df.columns
        assert "pnl" in df.columns
        assert "duration" in df.columns

    def test_empty_dataframe(self):
        """Test empty trade list to DataFrame."""
        df = TradeAnalyzer.trades_to_dataframe([])

        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)


class TestTradeStats:
    """Test TradeStats dataclass."""

    def test_to_dict(self, sample_trades):
        """Test TradeStats to_dict conversion."""
        stats = TradeAnalyzer.analyze(sample_trades)
        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert "total_trades" in stats_dict
        assert "win_rate" in stats_dict
        assert "profit_factor" in stats_dict
        assert stats_dict["total_trades"] == 5
