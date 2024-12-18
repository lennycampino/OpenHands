import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openhands.agenthub.securities_analysis_agent.agent import SecuritiesAnalysisAgent
from openhands.agenthub.securities_analysis_agent.models.security import (
    SecurityPosition,
    SecurityIdentifiers,
    SecurityType,
    AssetClass
)
from openhands.agenthub.securities_analysis_agent.models.report import (
    ReportType,
    ReportGenerationConfig
)

@pytest.fixture
def sample_security_position():
    """Create a sample security position."""
    return SecurityPosition(
        identifiers=SecurityIdentifiers(
            ticker="AAPL",
            isin="US0378331005"
        ),
        security_type=SecurityType.EQUITY,
        asset_class=AssetClass.EQUITY,
        quantity=1000,
        cost_basis=150.0,
        currency="USD",
        current_price=170.0,
        market_value=170000.0,
        unrealized_pnl=20000.0
    )

@pytest.fixture
def sample_portfolio(sample_security_position):
    """Create a sample portfolio."""
    return [
        sample_security_position,
        SecurityPosition(
            identifiers=SecurityIdentifiers(
                ticker="MSFT",
                isin="US5949181045"
            ),
            security_type=SecurityType.EQUITY,
            asset_class=AssetClass.EQUITY,
            quantity=500,
            cost_basis=250.0,
            currency="USD",
            current_price=280.0,
            market_value=140000.0,
            unrealized_pnl=15000.0
        )
    ]

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
        'Volume': np.random.randint(1000000, 5000000, 252)
    }, index=dates)

def test_end_to_end_analysis(
    mock_llm,
    mock_config,
    sample_portfolio,
    sample_market_data
):
    """Test end-to-end analysis workflow."""
    agent = SecuritiesAnalysisAgent(llm=mock_llm, config=mock_config)
    
    # Set test market data
    agent.test_market_data = sample_market_data
    
    # Test portfolio analysis
    analysis = agent.analyze_portfolio(sample_portfolio)
    assert analysis is not None
    assert len(analysis) == len(sample_portfolio)
    
    # Test report generation
    report = agent.generate_report(
        analysis,
        report_type=ReportType.PORTFOLIO_ANALYSIS
    )
    assert report is not None
    assert report.report_type == ReportType.PORTFOLIO_ANALYSIS
    assert len(report.sections) > 0
    
    # Verify risk metrics
    risk_metrics = agent.calculate_portfolio_risk(sample_portfolio)
    assert risk_metrics is not None
    assert risk_metrics.var_95 is not None
    assert risk_metrics.volatility is not None
    
    # Verify technical analysis
    technical = agent.analyze_technical_indicators(sample_market_data)
    assert technical is not None
    assert 'rsi' in technical.momentum_indicators
    assert len(technical.support_levels) > 0
    
    # Verify quantitative analysis
    quant = agent.perform_quantitative_analysis(sample_market_data['close'].pct_change().dropna())
    assert quant is not None
    assert 'sharpe_ratio' in quant
    assert 'max_drawdown' in quant

def test_data_validation(sample_market_data):
    """Test data validation capabilities."""
    from openhands.agenthub.securities_analysis_agent.data.processor import MarketDataProcessor
    
    processor = MarketDataProcessor()
    validation = processor.validate_data(sample_market_data)
    
    assert validation.is_valid
    assert validation.data_quality_score > 0.9
    assert len(validation.errors) == 0
    
    # Test with missing data
    bad_data = sample_market_data.copy()
    bad_data.loc[bad_data.index[0:10], 'Close'] = np.nan
    
    validation = processor.validate_data(bad_data)
    assert len(validation.warnings) > 0
    assert validation.data_quality_score < 0.9

def test_stress_testing(sample_portfolio):
    """Test stress testing capabilities."""
    from openhands.agenthub.securities_analysis_agent.analyzers.risk import RiskAnalyzer
    
    analyzer = RiskAnalyzer()
    scenarios = analyzer.get_default_stress_scenarios()
    
    # Create sample returns data
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.0001, 0.02, 252),
        'MSFT': np.random.normal(0.0001, 0.02, 252)
    })
    
    results = analyzer.run_stress_tests(
        sample_portfolio,
        returns,
        scenarios
    )
    
    assert len(results) == len(scenarios)
    for scenario_name, result in results.items():
        assert result.portfolio_impact is not None
        assert result.var_impact is not None
        assert len(result.position_impacts) == len(sample_portfolio)

def test_report_generation(sample_portfolio, sample_market_data):
    """Test report generation capabilities."""
    from openhands.agenthub.securities_analysis_agent.reports.generator import PDFReportGenerator
    from openhands.agenthub.securities_analysis_agent.models.report import (
        AnalysisReport,
        ExecutiveSummary,
        AnalysisSection,
        ConfidenceLevel
    )
    
    # Create sample report
    report = AnalysisReport(
        report_id="TEST001",
        report_type=ReportType.PORTFOLIO_ANALYSIS,
        timestamp=datetime.now(),
        title="Test Portfolio Analysis",
        executive_summary=ExecutiveSummary(
            key_findings=["Finding 1", "Finding 2"],
            recommendations=["Rec 1", "Rec 2"],
            risk_summary="Risk summary",
            opportunity_summary="Opportunity summary",
            confidence_level=ConfidenceLevel.HIGH,
            time_horizon="6 months"
        ),
        sections=[
            AnalysisSection(
                title="Analysis Section",
                content="Test content",
                charts=[],
                tables=[]
            )
        ]
    )
    
    generator = PDFReportGenerator()
    config = ReportGenerationConfig(
        template=None,
        output_format="pdf",
        include_charts=True
    )
    
    # Generate report
    generator.generate_report(report, config, "test_report.pdf")
    
    # Verify file was created
    import os
    assert os.path.exists("test_report.pdf")
    assert os.path.getsize("test_report.pdf") > 0