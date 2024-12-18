from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum

class SecurityType(Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    DERIVATIVE = "derivative"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"

class AssetClass(Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    ALTERNATIVES = "alternatives"
    CASH = "cash"
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"

@dataclass
class SecurityIdentifiers:
    """Identifiers for a security."""
    ticker: str
    isin: Optional[str] = None
    cusip: Optional[str] = None
    sedol: Optional[str] = None
    figi: Optional[str] = None

@dataclass
class SecurityPrice:
    """Price information for a security."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

@dataclass
class SecurityPosition:
    """A position in a security."""
    identifiers: SecurityIdentifiers
    security_type: SecurityType
    asset_class: AssetClass
    quantity: float
    cost_basis: float
    currency: str
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    position_date: datetime = datetime.now()

@dataclass
class SecurityFundamentals:
    """Fundamental data for a security."""
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    ebitda: Optional[float] = None
    fcf: Optional[float] = None

@dataclass
class SecurityAnalysis:
    """Complete analysis of a security."""
    position: SecurityPosition
    fundamentals: SecurityFundamentals
    technical_indicators: Dict[str, float]
    risk_metrics: Dict[str, float]
    price_history: List[SecurityPrice]
    analysis_date: datetime = datetime.now()
    analyst_notes: Optional[str] = None
    recommendation: Optional[str] = None
    target_price: Optional[float] = None
    confidence_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Union[float, str, Dict, List]]:
        """Convert the analysis to a dictionary format."""
        return {
            "ticker": self.position.identifiers.ticker,
            "security_type": self.position.security_type.value,
            "asset_class": self.position.asset_class.value,
            "current_price": self.position.current_price,
            "market_value": self.position.market_value,
            "unrealized_pnl": self.position.unrealized_pnl,
            "fundamentals": {
                k: v for k, v in self.fundamentals.__dict__.items()
                if v is not None
            },
            "technical_indicators": self.technical_indicators,
            "risk_metrics": self.risk_metrics,
            "recommendation": self.recommendation,
            "target_price": self.target_price,
            "confidence_score": self.confidence_score,
            "analysis_date": self.analysis_date.isoformat(),
            "analyst_notes": self.analyst_notes
        }