"""Event types flowing through the engine's queue.

The event loop is strictly causal: MARKET -> SIGNAL -> ORDER -> FILL.
A strategy can only ever react to information that has already been
emitted as a MarketEvent, which is the structural defense against
lookahead bias.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


@dataclass
class Event:
    type: EventType = field(init=False)


@dataclass
class MarketEvent(Event):
    """A new bar has arrived for one symbol."""
    timestamp: datetime
    symbol: str

    def __post_init__(self):
        self.type = EventType.MARKET


@dataclass
class SignalEvent(Event):
    """Strategy's desired exposure, in [-1, 1] of allocatable capital."""
    timestamp: datetime
    symbol: str
    target_weight: float

    def __post_init__(self):
        self.type = EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    """Portfolio-sized order in shares (signed). Executed on the NEXT bar."""
    timestamp: datetime
    symbol: str
    quantity: int  # signed: + buy, - sell

    def __post_init__(self):
        self.type = EventType.ORDER


@dataclass
class FillEvent(Event):
    """What the broker actually did to you."""
    timestamp: datetime
    symbol: str
    quantity: int          # signed, possibly smaller than ordered (capacity)
    fill_price: float      # includes slippage
    commission: float
    slippage_cost: float   # dollars lost vs. the reference (arrival) price

    def __post_init__(self):
        self.type = EventType.FILL
