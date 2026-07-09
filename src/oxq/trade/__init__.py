from oxq.trade.fees import FeeModel, PercentageFee, SideAwarePercentageFee
from oxq.trade.order_generator import PlannedOrder, generate_orders
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.trade.slippage import PercentageSlippage, SlippageModel

__all__ = [
    "FeeModel",
    "FillPriceMode",
    "PercentageFee",
    "PercentageSlippage",
    "PlannedOrder",
    "SideAwarePercentageFee",
    "SimBroker",
    "SlippageModel",
    "generate_orders",
]

# LiveBroker is only available when httpx + websockets are installed
try:
    from oxq.contrib.alpaca.client import AlpacaAPIError, AlpacaClient
    from oxq.trade.live_broker import LiveBroker

    __all__ += ["AlpacaAPIError", "AlpacaClient", "LiveBroker"]
except ImportError:
    pass
