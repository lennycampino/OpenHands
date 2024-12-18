import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Test analysis response"
    llm.completion.return_value = response
    return llm

@pytest.fixture
def mock_config():
    """Create a mock config."""
    config = Mock()
    config.max_iterations = 10
    return config

@pytest.fixture
def sample_financial_statements():
    """Create sample financial statement data."""
    # Income Statement
    income_statement = pd.DataFrame({
        'revenue': [100000, 120000, 150000],
        'cost_of_goods_sold': [60000, 70000, 85000],
        'gross_profit': [40000, 50000, 65000],
        'operating_expenses': [20000, 25000, 30000],
        'operating_income': [20000, 25000, 35000],
        'net_income': [15000, 19000, 26000],
        'ebitda': [25000, 31000, 42000],
        'interest_expense': [2000, 2500, 3000],
        'income_tax': [3000, 3500, 6000],
        'depreciation': [4000, 4500, 5000],
        'amortization': [1000, 1500, 2000]
    }, index=pd.date_range(end=datetime.now(), periods=3, freq='Q'))

    # Balance Sheet
    balance_sheet = pd.DataFrame({
        'cash': [10000, 12000, 15000],
        'accounts_receivable': [15000, 18000, 22000],
        'inventory': [20000, 25000, 30000],
        'total_current_assets': [45000, 55000, 67000],
        'total_assets': [100000, 120000, 150000],
        'accounts_payable': [10000, 12000, 15000],
        'short_term_debt': [15000, 18000, 20000],
        'total_current_liabilities': [25000, 30000, 35000],
        'long_term_debt': [30000, 35000, 40000],
        'total_liabilities': [55000, 65000, 75000],
        'total_equity': [45000, 55000, 75000]
    }, index=pd.date_range(end=datetime.now(), periods=3, freq='Q'))

    # Cash Flow Statement
    cash_flow = pd.DataFrame({
        'operating_cash_flow': [18000, 22000, 28000],
        'capital_expenditures': [-8000, -10000, -12000],
        'free_cash_flow': [10000, 12000, 16000],
        'debt_issuance': [5000, 6000, 7000],
        'debt_repayment': [-4000, -5000, -6000],
        'dividends_paid': [-2000, -2500, -3000]
    }, index=pd.date_range(end=datetime.now(), periods=3, freq='Q'))

    return {
        'income_statement': income_statement,
        'balance_sheet': balance_sheet,
        'cash_flow': cash_flow
    }

@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    np.random.seed(42)
    
    return pd.DataFrame({
        'Open': np.random.normal(100, 2, 252),
        'High': np.random.normal(101, 2, 252),
        'Low': np.random.normal(99, 2, 252),
        'Close': np.random.normal(100, 2, 252),
        'Volume': np.random.randint(1000000, 5000000, 252),
        'Adj Close': np.random.normal(100, 2, 252)
    }, index=dates)

@pytest.fixture
def sample_factor_data():
    """Create sample factor data."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    np.random.seed(42)
    
    return pd.DataFrame({
        'Market': np.random.normal(0.0001, 0.01, 252),
        'SMB': np.random.normal(0.0001, 0.005, 252),
        'HML': np.random.normal(0.0001, 0.005, 252),
        'Momentum': np.random.normal(0.0001, 0.006, 252),
        'Quality': np.random.normal(0.0001, 0.004, 252)
    }, index=dates)

@pytest.fixture
def sample_industry_data():
    """Create sample industry metrics."""
    return {
        'avg_pe_ratio': 15.5,
        'avg_pb_ratio': 2.3,
        'avg_profit_margin': 0.15,
        'avg_roe': 0.12,
        'avg_debt_to_equity': 1.2,
        'revenue_growth': 0.08,
        'market_size': 1000000000000,
        'competition_level': 0.75
    }

@pytest.fixture
def mock_yfinance():
    """Create a mock yfinance Ticker object."""
    mock_ticker = MagicMock()
    
    # Mock history method
    mock_ticker.history.return_value = pd.DataFrame({
        'Open': np.random.normal(100, 2, 252),
        'High': np.random.normal(101, 2, 252),
        'Low': np.random.normal(99, 2, 252),
        'Close': np.random.normal(100, 2, 252),
        'Volume': np.random.randint(1000000, 5000000, 252),
        'Adj Close': np.random.normal(100, 2, 252)
    }, index=pd.date_range(end=datetime.now(), periods=252, freq='B'))
    
    # Mock info property
    mock_ticker.info = {
        'marketCap': 2000000000000,
        'trailingPE': 25.5,
        'priceToBook': 8.5,
        'dividendYield': 0.015,
        'debtToEquity': 1.5,
        'returnOnEquity': 0.35,
        'returnOnAssets': 0.15,
        'revenueGrowth': 0.25,
        'operatingMargins': 0.30,
        'industry': 'Technology'
    }
    
    return mock_ticker

@pytest.fixture
def mock_market_data_processor(sample_market_data):
    """Create a mock MarketDataProcessor."""
    processor = Mock()
    processor.fetch_market_data.return_value = sample_market_data
    processor.validate_data.return_value = Mock(
        is_valid=True,
        errors=[],
        warnings=[],
        missing_fields=[],
        data_quality_score=0.95
    )
    return processor

@pytest.fixture
def mock_risk_analyzer():
    """Create a mock RiskAnalyzer."""
    analyzer = Mock()
    analyzer.calculate_risk_metrics.return_value = Mock(
        var_95=0.02,
        var_99=0.03,
        cvar_95=0.025,
        cvar_99=0.035,
        expected_shortfall=0.025,
        volatility=0.15,
        downside_volatility=0.12,
        correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
        liquidity_score=0.8,
        concentration_score=0.3
    )
    return analyzer

@pytest.fixture
def mock_technical_analyzer():
    """Create a mock TechnicalAnalyzer."""
    analyzer = Mock()
    analyzer.analyze.return_value = Mock(
        moving_averages={'sma_20': 100, 'sma_50': 98, 'sma_200': 95},
        rsi=55.0,
        macd={'line': 0.5, 'signal': 0.3, 'histogram': 0.2},
        bollinger_bands={'upper': 102, 'lower': 98, 'width': 4},
        atr=2.0,
        volume_metrics={'obv': 1000000, 'volume_sma': 500000},
        momentum_indicators={'rsi': 55, 'stoch_k': 65, 'stoch_d': 60},
        support_levels=[95, 97],
        resistance_levels=[103, 105],
        trend_strength=0.7
    )
    return analyzer