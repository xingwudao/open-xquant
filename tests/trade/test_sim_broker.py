"""Tests for SimBroker."""

from decimal import Decimal

import pandas as pd
import pytest

from oxq.core.types import Broker, FillReceiver, Order, OrderRouter
from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.trade.slippage import PercentageSlippage


def test_sim_broker_satisfies_order_router_protocol() -> None:
    assert isinstance(SimBroker(), OrderRouter)


def test_sim_broker_satisfies_fill_receiver_protocol() -> None:
    assert isinstance(SimBroker(), FillReceiver)


def test_sim_broker_satisfies_broker_protocol() -> None:
    assert isinstance(SimBroker(), Broker)


def test_submit_order_returns_id() -> None:
    broker = SimBroker()
    order = Order(symbol="AAPL", side="BUY", shares=100)
    order_id = broker.submit_order(order)
    assert isinstance(order_id, str)
    assert len(order_id) > 0


def test_fill_market_orders() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=2)
    mktdata = {
        "AAPL": pd.DataFrame({"close": [150.0, 155.0]}, index=dates),
    }
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].filled_price == Decimal("150")
    assert fills[0].order.symbol == "AAPL"


def test_get_fills_clears_after_read() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {
        "AAPL": pd.DataFrame({"close": [150.0]}, index=dates),
    }
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    assert len(broker.get_fills()) == 1
    assert len(broker.get_fills()) == 0


def test_multi_symbol_fill() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {
        "AAPL": pd.DataFrame({"close": [150.0]}, index=dates),
        "MSFT": pd.DataFrame({"close": [300.0]}, index=dates),
    }
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.submit_order(Order(symbol="MSFT", side="BUY", shares=50))
    broker.fill_pending_orders(mktdata, dates[0])

    fills = broker.get_fills()
    assert len(fills) == 2
    prices = {f.order.symbol: f.filled_price for f in fills}
    assert prices["AAPL"] == Decimal("150")
    assert prices["MSFT"] == Decimal("300")


def test_fee_model() -> None:
    broker = SimBroker(fee_model=PercentageFee(rate=Decimal("0.001"), min_fee=Decimal("5")))
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {"AAPL": pd.DataFrame({"close": [150.0]}, index=dates)}
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    fill = broker.get_fills()[0]
    assert fill.fee == Decimal("15.0")  # 150 * 100 * 0.001


def test_slippage_model() -> None:
    broker = SimBroker(slippage_model=PercentageSlippage(rate=Decimal("0.01")))
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {"AAPL": pd.DataFrame({"close": [100.0]}, index=dates)}
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    fill = broker.get_fills()[0]
    assert fill.filled_price == Decimal("100") * (1 + Decimal("0.01"))


def test_stop_order_triggers() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=3)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [150.0, 145.0, 138.0]}, index=dates,
        ),
    }
    # Place stop at 140
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("140"),
    ))

    # Day 1: close=150 > 140, no trigger
    broker.process_pending_orders(mktdata, dates[0])
    broker.fill_market_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0

    # Day 2: close=145 > 140, no trigger
    broker.process_pending_orders(mktdata, dates[1])
    broker.fill_market_orders(mktdata, dates[1])
    assert len(broker.get_fills()) == 0

    # Day 3: close=138 <= 140, triggers
    broker.process_pending_orders(mktdata, dates[2])
    broker.fill_market_orders(mktdata, dates[2])
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].order.order_type == "stop"
    assert fills[0].filled_price == Decimal("140")


def test_limit_order_triggers() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=3)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [150.0, 160.0, 185.0]}, index=dates,
        ),
    }
    # Place limit sell at 180
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="limit", limit_price=Decimal("180"),
    ))

    # Day 1: 150 < 180, no trigger
    broker.process_pending_orders(mktdata, dates[0])
    broker.fill_market_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0

    # Day 3: 185 >= 180, triggers at limit price
    broker.process_pending_orders(mktdata, dates[2])
    broker.fill_market_orders(mktdata, dates[2])
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].filled_price == Decimal("180")


def test_trailing_stop_order() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=4)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [100.0, 110.0, 105.0, 103.0]}, index=dates,
        ),
    }
    # Trail 5% from high
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="trailing_stop", trail_pct=0.05,
    ))

    # Day 1: HWM=100, stop=95, close=100, no trigger
    broker.process_pending_orders(mktdata, dates[0])
    broker.fill_market_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0

    # Day 2: HWM=110, stop=104.5, close=110, no trigger
    broker.process_pending_orders(mktdata, dates[1])
    broker.fill_market_orders(mktdata, dates[1])
    assert len(broker.get_fills()) == 0

    # Day 3: HWM=110, stop=104.5, close=105, no trigger
    broker.process_pending_orders(mktdata, dates[2])
    broker.fill_market_orders(mktdata, dates[2])
    assert len(broker.get_fills()) == 0

    # Day 4: HWM=110, stop=104.5, close=103 <= 104.5, triggers
    broker.process_pending_orders(mktdata, dates[3])
    broker.fill_market_orders(mktdata, dates[3])
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].order.order_type == "trailing_stop"


def test_stop_dedup_replaces_old() -> None:
    broker = SimBroker()

    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("140"),
    ))
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("145"),
    ))

    open_orders = broker.get_open_orders(symbol="AAPL")
    assert len(open_orders) == 1
    assert open_orders[0].order.stop_price == Decimal("145")


def test_get_open_orders_filter_by_symbol() -> None:
    broker = SimBroker()
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("140"),
    ))
    broker.submit_order(Order(
        symbol="MSFT", side="SELL", shares=50,
        order_type="stop", stop_price=Decimal("280"),
    ))

    assert len(broker.get_open_orders(symbol="AAPL")) == 1
    assert len(broker.get_open_orders(symbol="MSFT")) == 1
    assert len(broker.get_open_orders()) == 2


def test_buy_stop_order_triggers() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=3)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [150.0, 155.0, 165.0]}, index=dates,
        ),
    }
    broker.submit_order(Order(
        symbol="AAPL", side="BUY", shares=100,
        order_type="stop", stop_price=Decimal("160"),
    ))

    # Day 1: close=150 < 160, no trigger
    broker.process_pending_orders(mktdata, dates[0])
    broker.fill_market_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0

    # Day 2: close=155 < 160, no trigger
    broker.process_pending_orders(mktdata, dates[1])
    broker.fill_market_orders(mktdata, dates[1])
    assert len(broker.get_fills()) == 0

    # Day 3: close=165 >= 160, triggers
    broker.process_pending_orders(mktdata, dates[2])
    broker.fill_market_orders(mktdata, dates[2])
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].filled_price == Decimal("160")
    assert fills[0].order.side == "BUY"


def test_buy_limit_order_triggers() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=3)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [150.0, 145.0, 135.0]}, index=dates,
        ),
    }
    broker.submit_order(Order(
        symbol="AAPL", side="BUY", shares=100,
        order_type="limit", limit_price=Decimal("140"),
    ))

    # Day 1: close=150 > 140, no trigger
    broker.process_pending_orders(mktdata, dates[0])
    broker.fill_market_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0

    # Day 2: close=145 > 140, no trigger
    broker.process_pending_orders(mktdata, dates[1])
    broker.fill_market_orders(mktdata, dates[1])
    assert len(broker.get_fills()) == 0

    # Day 3: close=135 <= 140, triggers
    broker.process_pending_orders(mktdata, dates[2])
    broker.fill_market_orders(mktdata, dates[2])
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].filled_price == Decimal("140")
    assert fills[0].order.side == "BUY"


def test_stop_order_with_slippage() -> None:
    broker = SimBroker(slippage_model=PercentageSlippage(rate=Decimal("0.01")))
    dates = pd.bdate_range("2024-01-01", periods=2)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [150.0, 135.0]}, index=dates,
        ),
    }
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("140"),
    ))

    # Day 1: no trigger
    broker.process_pending_orders(mktdata, dates[0])
    broker.fill_market_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0

    # Day 2: close=135 <= 140, triggers
    broker.process_pending_orders(mktdata, dates[1])
    broker.fill_market_orders(mktdata, dates[1])
    fills = broker.get_fills()
    assert len(fills) == 1
    # SELL slippage: 140 * (1 - 0.01) = 138.6
    assert fills[0].filled_price == Decimal("140") * (1 - Decimal("0.01"))


def test_process_pending_skips_missing_symbol() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {
        "MSFT": pd.DataFrame({"close": [300.0]}, index=dates),
    }
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("140"),
    ))

    # Should not crash; no fills
    broker.process_pending_orders(mktdata, dates[0])
    broker.fill_market_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0


def test_process_pending_skips_missing_date() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=1)
    other_date = pd.Timestamp("2024-06-01")
    mktdata = {
        "AAPL": pd.DataFrame({"close": [150.0]}, index=dates),
    }
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("140"),
    ))

    # Should not crash; no fills
    broker.process_pending_orders(mktdata, other_date)
    broker.fill_market_orders(mktdata, other_date)
    assert len(broker.get_fills()) == 0


def test_no_fee_model_returns_zero_fee() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {"AAPL": pd.DataFrame({"close": [150.0]}, index=dates)}
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    fill = broker.get_fills()[0]
    assert fill.fee == Decimal("0")


def test_fee_and_slippage_combined() -> None:
    broker = SimBroker(
        fee_model=PercentageFee(rate=Decimal("0.001"), min_fee=Decimal("0")),
        slippage_model=PercentageSlippage(rate=Decimal("0.01")),
    )
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {"AAPL": pd.DataFrame({"close": [100.0]}, index=dates)}
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    fill = broker.get_fills()[0]
    # slippage first: 100 * 1.01 = 101
    assert fill.filled_price == Decimal("100") * (1 + Decimal("0.01"))
    # fee on slipped price: 101 * 100 * 0.001 = 10.1
    expected_fee = Decimal("101") * 100 * Decimal("0.001")
    assert fill.fee == expected_fee


def test_fill_pending_orders_legacy() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=2)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [150.0, 135.0]}, index=dates,
        ),
    }
    # Submit a market order
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    # Submit a stop order that will trigger on day 2
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=50,
        order_type="stop", stop_price=Decimal("140"),
    ))

    # Day 1: market order fills, stop does not trigger (150 > 140)
    broker.fill_pending_orders(mktdata, dates[0])
    fills_day1 = broker.get_fills()
    assert len(fills_day1) == 1
    assert fills_day1[0].order.side == "BUY"
    assert fills_day1[0].filled_price == Decimal("150")

    # Day 2: stop triggers (135 <= 140)
    broker.fill_pending_orders(mktdata, dates[1])
    fills_day2 = broker.get_fills()
    assert len(fills_day2) == 1
    assert fills_day2[0].order.side == "SELL"
    assert fills_day2[0].order.order_type == "stop"
    assert fills_day2[0].filled_price == Decimal("140")


def test_cap_pending_sells() -> None:
    """cap_pending_sells should reduce shares on SELL orders."""
    broker = SimBroker()
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=500,
        order_type="stop", stop_price=Decimal("140"),
    ))
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=500,
        order_type="limit", limit_price=Decimal("180"),
    ))

    broker.cap_pending_sells("AAPL", max_shares=300)

    for m in broker.get_open_orders("AAPL"):
        assert m.order.shares == 300


def test_cap_pending_sells_cancel_on_zero() -> None:
    """cap_pending_sells with max_shares=0 should cancel orders."""
    broker = SimBroker()
    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=500,
        order_type="stop", stop_price=Decimal("140"),
    ))
    broker.cap_pending_sells("AAPL", max_shares=0)
    assert len(broker.get_open_orders("AAPL")) == 0


def test_cap_pending_sells_then_trigger() -> None:
    """After capping, a triggered order should fill with the capped shares."""
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=2)
    mktdata = {"AAPL": pd.DataFrame({"close": [150.0, 135.0]}, index=dates)}

    broker.submit_order(Order(
        symbol="AAPL", side="SELL", shares=500,
        order_type="stop", stop_price=Decimal("140"),
    ))
    # Position was reduced to 300, cap the order
    broker.cap_pending_sells("AAPL", max_shares=300)

    # Day 1: no trigger
    broker.process_pending_orders(mktdata, dates[0])
    assert len(broker.get_fills()) == 0

    # Day 2: triggers with capped shares
    broker.process_pending_orders(mktdata, dates[1])
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].order.shares == 300


def test_market_order_marked_filled_in_orderbook() -> None:
    """After fill_market_orders, market orders should no longer appear open."""
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {"AAPL": pd.DataFrame({"close": [150.0]}, index=dates)}

    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    # Before fill: order is open in OrderBook
    assert len(broker.get_open_orders()) == 1

    broker.fill_market_orders(mktdata, dates[0])
    # After fill: order must be marked filled, not open
    assert len(broker.get_open_orders()) == 0

    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].filled_price == Decimal("150")


def test_market_orders_do_not_accumulate() -> None:
    """Multiple bars of market orders should not leave garbage in OrderBook."""
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=3)
    mktdata = {"AAPL": pd.DataFrame({"close": [100.0, 110.0, 120.0]}, index=dates)}

    for i, d in enumerate(dates):
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=50))
        broker.fill_market_orders(mktdata, d)
        broker.get_fills()  # drain fills
        # No open orders should remain after each fill
        assert len(broker.get_open_orders()) == 0, f"Open orders remain after bar {i}"


class TestFillPriceMode:
    def _make_mktdata(self):
        """Two-bar dataset: bar0 close=100, bar1 open=102 high=105 low=98 close=101."""
        dates = pd.date_range("2024-01-01", periods=2)
        df = pd.DataFrame({
            "open": [100.0, 102.0],
            "high": [101.0, 105.0],
            "low": [99.0, 98.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        }, index=dates)
        return {"AAPL": df}, dates

    def test_close_mode(self):
        """Default mode fills at current bar close."""
        mktdata, dates = self._make_mktdata()
        broker = SimBroker(fill_price_mode=FillPriceMode.CLOSE)
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))
        broker.on_bar_close(mktdata, dates[0])
        fills = broker.get_fills()
        assert fills[0].filled_price == Decimal("100")

    def test_next_open_mode(self):
        """NEXT_OPEN fills at next bar's open price."""
        mktdata, dates = self._make_mktdata()
        broker = SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS")
        broker.set_current_date(dates[0])
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))
        broker.on_bar_close(mktdata, dates[0])
        assert broker.get_fills() == []
        broker.on_bar_open(mktdata, dates[1])
        fills = broker.get_fills()
        assert fills[0].filled_price == Decimal("102")
        assert fills[0].filled_at == dates[1].isoformat()

    def test_next_open_preserves_positive_timezone_session_date(self):
        """NEXT_OPEN should preserve local session dates for tz-aware bars."""
        dates = pd.DatetimeIndex(
            [
                pd.Timestamp("2025-02-07 00:00:00+08:00"),
                pd.Timestamp("2025-02-10 00:00:00+08:00"),
            ]
        )
        mktdata = {
            "510300.SS": pd.DataFrame(
                {
                    "open": [100.0, 101.0],
                    "high": [100.0, 101.0],
                    "low": [100.0, 101.0],
                    "close": [100.0, 101.0],
                    "volume": [1000, 1000],
                },
                index=dates,
            )
        }
        broker = SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XSHG")
        broker.set_current_date(dates[0])
        broker.submit_order(Order(symbol="510300.SS", side="BUY", shares=10))

        broker.on_bar_close(mktdata, dates[0])
        broker.on_bar_open(mktdata, dates[1])

        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].filled_price == Decimal("101.0")
        assert fills[0].filled_at == dates[1].isoformat()
        assert broker.get_all_orders()[0].status == "filled"

    def test_next_open_without_current_date_uses_close_bar_as_submission_time(self):
        mktdata, dates = self._make_mktdata()
        broker = SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS")
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))

        broker.on_bar_close(mktdata, dates[0])
        assert broker.get_fills() == []
        broker.on_bar_open(mktdata, dates[1])

        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].filled_price == Decimal("102")
        assert fills[0].filled_at == dates[1].isoformat()

    def test_next_high_mode_is_rejected(self):
        """NEXT_HIGH is not causal for market-order backtests."""
        with pytest.raises(ValueError, match="NEXT_HIGH"):
            SimBroker(fill_price_mode=FillPriceMode.NEXT_HIGH)

    def test_next_low_mode_is_rejected(self):
        """NEXT_LOW is not causal for market-order backtests."""
        with pytest.raises(ValueError, match="NEXT_LOW"):
            SimBroker(fill_price_mode=FillPriceMode.NEXT_LOW)

    def test_next_open_requires_market_calendar(self):
        with pytest.raises(ValueError, match="market_calendar"):
            SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN)

    @pytest.mark.parametrize(
        ("mode", "expected_price"),
        [
            (FillPriceMode.NEXT_CLOSE, Decimal("101")),
            (FillPriceMode.NEXT_MID, Decimal("101.5")),
            (FillPriceMode.NEXT_AVG, Decimal("101.5")),
            (FillPriceMode.NEXT_HL2, Decimal("101.5")),
        ],
    )
    def test_next_session_modes_fill_on_next_session_not_same_bar(self, mode, expected_price):
        mktdata, dates = self._make_mktdata()
        broker = SimBroker(fill_price_mode=mode, market_calendar="XNYS")
        broker.set_current_date(dates[0])
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))

        broker.on_bar_close(mktdata, dates[0])
        assert broker.get_fills() == []
        broker.on_bar_open(mktdata, dates[1])
        assert broker.get_fills() == []
        broker.on_bar_close(mktdata, dates[1])

        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].filled_price == expected_price
        assert fills[0].filled_at == dates[1].isoformat()

    @pytest.mark.parametrize(
        ("mode", "expected_price"),
        [
            (FillPriceMode.NEXT_CLOSE, Decimal("101")),
            (FillPriceMode.NEXT_MID, Decimal("101.5")),
            (FillPriceMode.NEXT_AVG, Decimal("101.5")),
            (FillPriceMode.NEXT_HL2, Decimal("101.5")),
        ],
    )
    def test_next_session_close_modes_do_not_fill_via_due_open_hook(self, mode, expected_price):
        mktdata, dates = self._make_mktdata()
        broker = SimBroker(fill_price_mode=mode, market_calendar="XNYS")
        broker.set_current_date(dates[0])
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))

        broker.fill_due_market_orders(mktdata, dates[1])
        assert broker.get_fills() == []

        broker.on_bar_close(mktdata, dates[1])

        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].filled_price == expected_price
        assert fills[0].filled_at == dates[1].isoformat()

    def test_next_open_still_fills_via_due_open_hook(self):
        mktdata, dates = self._make_mktdata()
        broker = SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS")
        broker.set_current_date(dates[0])
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))

        broker.fill_due_market_orders(mktdata, dates[1])

        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].filled_price == Decimal("102")
        assert fills[0].filled_at == dates[1].isoformat()

    @pytest.mark.parametrize(
        "mode",
        [FillPriceMode.NEXT_CLOSE, FillPriceMode.NEXT_MID, FillPriceMode.NEXT_AVG, FillPriceMode.NEXT_HL2],
    )
    def test_all_next_session_modes_require_market_calendar(self, mode):
        with pytest.raises(ValueError, match="market_calendar"):
            SimBroker(fill_price_mode=mode)

    def test_next_hl2_uses_next_high_low_midpoint_not_open_close_midpoint(self):
        dates = pd.bdate_range("2024-01-01", periods=2)
        mktdata = {
            "AAPL": pd.DataFrame(
                {
                    "open": [100.0, 102.0],
                    "high": [103.0, 110.0],
                    "low": [99.0, 90.0],
                    "close": [101.0, 101.0],
                },
                index=dates,
            )
        }
        broker = SimBroker(fill_price_mode=FillPriceMode.NEXT_HL2, market_calendar="XNYS")
        broker.set_current_date(dates[0])
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))

        broker.on_bar_close(mktdata, dates[0])
        broker.on_bar_close(mktdata, dates[1])

        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].filled_price == Decimal("100")

    def test_next_mode_without_submission_timestamp_does_not_peek(self):
        """NEXT_OPEN without created_at should not read next-bar prices."""
        mktdata, dates = self._make_mktdata()
        broker = SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS")
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))
        broker.on_bar_open(mktdata, dates[0])
        broker.on_bar_close(mktdata, dates[0])
        assert broker.get_fills() == []

        broker.on_bar_open(mktdata, dates[1])
        fills = broker.get_fills()
        assert fills[0].filled_price == Decimal("102")

    def test_next_mode_last_bar_skips(self):
        """NEXT_OPEN on last bar -> no fill (no next bar)."""
        mktdata, dates = self._make_mktdata()
        broker = SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS")
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))
        broker.on_bar_close(mktdata, dates[1])  # last bar
        fills = broker.get_fills()
        assert len(fills) == 0

    def test_next_open_expires_when_next_bar_is_missing(self):
        """NEXT_OPEN orders should not fill on a later stale bar."""
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame(
            {
                "open": [100.0, 104.0],
                "high": [101.0, 105.0],
                "low": [99.0, 103.0],
                "close": [100.0, 104.0],
                "volume": [1000, 1000],
            },
            index=[dates[0], dates[2]],
        )
        mktdata = {"AAPL": df}
        broker = SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS")
        broker.set_current_date(dates[0])
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))

        broker.on_bar_close(mktdata, dates[0])
        broker.on_bar_open(mktdata, dates[2])

        assert broker.get_fills() == []
        assert broker.get_open_orders() == []

    def test_next_open_multiple_market_orders_fill_independently(self):
        """Broker-level market orders should not replace each other."""
        mktdata, dates = self._make_mktdata()
        broker = SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS")
        broker.set_current_date(dates[0])
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=10))
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=20))

        broker.on_bar_close(mktdata, dates[0])
        broker.on_bar_open(mktdata, dates[1])

        fills = broker.get_fills()
        orders = broker.get_all_orders()
        assert [fill.order.shares for fill in fills] == [10, 20]
        assert [order.status for order in orders] == ["filled", "filled"]


class TestFillPriceModeMid:
    def test_mid_price_fill(self):
        """MID mode fills at (open + close) / 2."""
        broker = SimBroker(fill_price_mode=FillPriceMode.MID)
        order = Order(symbol="A", side="BUY", shares=100)
        broker.submit_order(order)

        df = pd.DataFrame(
            {"open": [10.0], "high": [12.0], "low": [9.0], "close": [11.0]},
            index=pd.to_datetime(["2025-01-06"]),
        )
        mktdata = {"A": df}
        date = pd.Timestamp("2025-01-06")

        broker.on_bar_close(mktdata, date)
        fills = broker.get_fills()

        assert len(fills) == 1
        # mid = (10.0 + 11.0) / 2 = 10.5
        assert fills[0].filled_price == Decimal("10.5")


class TestNaNAndInfinityPrices:
    """close price may be NaN or Infinity — broker must skip, not crash."""

    def test_stop_order_nan_close_skips(self):
        """NaN close should not trigger stop order or raise InvalidOperation."""
        broker = SimBroker()
        dates = pd.bdate_range("2024-01-01", periods=2)
        mktdata = {
            "AAPL": pd.DataFrame(
                {"close": [float("nan"), 135.0]}, index=dates,
            ),
        }
        broker.submit_order(Order(
            symbol="AAPL", side="SELL", shares=100,
            order_type="stop", stop_price=Decimal("140"),
        ))
        # Day 1: close=NaN, should skip gracefully (no fill, no crash)
        broker.process_pending_orders(mktdata, dates[0])
        assert len(broker.get_fills()) == 0
        # Order still open
        assert len(broker.get_open_orders()) == 1

        # Day 2: close=135 <= 140, triggers normally
        broker.process_pending_orders(mktdata, dates[1])
        fills = broker.get_fills()
        assert len(fills) == 1

    def test_limit_order_nan_close_skips(self):
        """NaN close should not trigger limit order or raise InvalidOperation."""
        broker = SimBroker()
        dates = pd.bdate_range("2024-01-01", periods=1)
        mktdata = {
            "AAPL": pd.DataFrame({"close": [float("nan")]}, index=dates),
        }
        broker.submit_order(Order(
            symbol="AAPL", side="SELL", shares=100,
            order_type="limit", limit_price=Decimal("180"),
        ))
        broker.process_pending_orders(mktdata, dates[0])
        assert len(broker.get_fills()) == 0
        assert len(broker.get_open_orders()) == 1

    def test_trailing_stop_nan_close_skips(self):
        """NaN close should not trigger trailing stop or corrupt high-water mark."""
        broker = SimBroker()
        dates = pd.bdate_range("2024-01-01", periods=2)
        mktdata = {
            "AAPL": pd.DataFrame(
                {"close": [float("nan"), 100.0]}, index=dates,
            ),
        }
        broker.submit_order(Order(
            symbol="AAPL", side="SELL", shares=100,
            order_type="trailing_stop", trail_pct=0.05,
        ))
        # Day 1: NaN should be skipped, high-water mark should NOT be set
        broker.process_pending_orders(mktdata, dates[0])
        assert len(broker.get_fills()) == 0

        # Day 2: valid price, should initialize high-water mark normally
        broker.process_pending_orders(mktdata, dates[1])
        assert len(broker.get_fills()) == 0
        managed = broker.get_open_orders()[0]
        assert managed.trail_high_water == Decimal("100")

    def test_market_order_nan_close_skips(self):
        """NaN close should not fill market order."""
        broker = SimBroker()
        dates = pd.bdate_range("2024-01-01", periods=2)
        mktdata = {
            "AAPL": pd.DataFrame(
                {"close": [float("nan"), 150.0]}, index=dates,
            ),
        }
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
        # Day 1: NaN close, should not fill
        broker.fill_market_orders(mktdata, dates[0])
        assert len(broker.get_fills()) == 0
        # Order still pending, fills on day 2
        broker.fill_market_orders(mktdata, dates[1])
        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].filled_price == Decimal("150")

    def test_stop_order_inf_close_skips(self):
        """Infinity close should not trigger stop order."""
        broker = SimBroker()
        dates = pd.bdate_range("2024-01-01", periods=1)
        mktdata = {
            "AAPL": pd.DataFrame({"close": [float("inf")]}, index=dates),
        }
        broker.submit_order(Order(
            symbol="AAPL", side="SELL", shares=100,
            order_type="stop", stop_price=Decimal("140"),
        ))
        broker.process_pending_orders(mktdata, dates[0])
        assert len(broker.get_fills()) == 0
        assert len(broker.get_open_orders()) == 1
