import pytest
import numpy as np
from datetime import datetime, timedelta

from openhands.agenthub.aifmd_compliance_agent.portfolio_risk_manager import (
    PortfolioPosition,
    RiskMetrics
)

@pytest.fixture
def sample_risk_metrics():
    """Create sample risk metrics for testing."""
    return RiskMetrics(
        var_95=0.02,
        var_99=0.03,
        expected_shortfall=0.025,
        volatility=0.15,
        sharpe_ratio=1.2,
        tracking_error=0.03,
        information_ratio=0.8,
        beta=1.1,
        alpha=0.02
    )

@pytest.fixture
def sample_portfolio_positions():
    """Create a diverse set of portfolio positions for testing."""
    return [
        # Equity positions
        PortfolioPosition(
            asset_id="EQUITY_TECH_1",
            asset_type="equity",
            quantity=5000,
            market_value=500000.0,
            currency="EUR",
            sector="Technology",
            geography="Europe",
            rating=None,
            leverage=None,
            counterparty=None
        ),
        PortfolioPosition(
            asset_id="EQUITY_FIN_1",
            asset_type="equity",
            quantity=3000,
            market_value=300000.0,
            currency="EUR",
            sector="Finance",
            geography="Europe",
            rating=None,
            leverage=None,
            counterparty=None
        ),
        # Bond positions
        PortfolioPosition(
            asset_id="BOND_GOV_1",
            asset_type="government_bond",
            quantity=1000000,
            market_value=1000000.0,
            currency="EUR",
            sector="Government",
            geography="Europe",
            rating="AAA",
            leverage=None,
            counterparty="German Government"
        ),
        PortfolioPosition(
            asset_id="BOND_CORP_1",
            asset_type="corporate_bond",
            quantity=500000,
            market_value=500000.0,
            currency="EUR",
            sector="Industrial",
            geography="Europe",
            rating="A",
            leverage=None,
            counterparty="Corp A"
        ),
        # Alternative investments
        PortfolioPosition(
            asset_id="PE_FUND_1",
            asset_type="private_equity",
            quantity=1,
            market_value=250000.0,
            currency="EUR",
            sector="Various",
            geography="Europe",
            rating=None,
            leverage=1.5,
            counterparty="PE Fund A"
        ),
        PortfolioPosition(
            asset_id="HEDGE_FUND_1",
            asset_type="hedge_fund",
            quantity=1,
            market_value=450000.0,
            currency="EUR",
            sector="Various",
            geography="Global",
            rating=None,
            leverage=2.0,
            counterparty="Hedge Fund B"
        )
    ]

@pytest.fixture
def sample_returns_data():
    """Create sample returns data for testing."""
    np.random.seed(42)
    
    # Generate one year of daily returns
    dates = [datetime.now() - timedelta(days=x) for x in range(252)]
    
    return {
        "dates": dates,
        "portfolio_returns": np.random.normal(0.0001, 0.02, 252),
        "benchmark_returns": np.random.normal(0.0001, 0.018, 252),
        "market_returns": np.random.normal(0.0001, 0.015, 252)
    }

@pytest.fixture
def sample_concentration_data():
    """Create sample concentration data for testing."""
    return {
        "asset_type": {
            "equity": 0.35,
            "government_bond": 0.30,
            "corporate_bond": 0.15,
            "private_equity": 0.10,
            "hedge_fund": 0.10
        },
        "sector": {
            "Technology": 0.20,
            "Finance": 0.15,
            "Government": 0.30,
            "Industrial": 0.15,
            "Various": 0.20
        },
        "geography": {
            "Europe": 0.85,
            "Global": 0.15
        },
        "counterparty": {
            "German Government": 0.30,
            "Corp A": 0.15,
            "PE Fund A": 0.10,
            "Hedge Fund B": 0.10,
            "None": 0.35
        }
    }

@pytest.fixture
def sample_leverage_data():
    """Create sample leverage metrics for testing."""
    return {
        "gross_leverage": 1.8,
        "net_leverage": 1.2,
        "long_exposure": 1.5,
        "short_exposure": 0.3
    }

@pytest.fixture
def sample_liquidity_data():
    """Create sample liquidity assessment data for testing."""
    return {
        "liquidity_profile": {
            "highly_liquid": 0.45,
            "moderately_liquid": 0.25,
            "less_liquid": 0.20,
            "illiquid": 0.10
        },
        "liquidity_coverage_ratio": 1.8,
        "days_to_liquidate_90_percent": 5.0
    }