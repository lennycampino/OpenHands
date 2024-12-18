import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from openhands.agenthub.aifmd_compliance_agent.agent import AIFMDComplianceAgent
from openhands.agenthub.aifmd_compliance_agent.portfolio_risk_manager import (
    PortfolioPosition,
    RiskMetrics,
    PortfolioRiskManager
)
from openhands.controller.state.state import State
from openhands.events.action import (
    Action,
    AgentFinishAction,
    AgentRejectAction,
    BrowseURLAction,
    FileReadAction,
    FileWriteAction,
    MessageAction,
)
from openhands.events.observation import BrowserOutputObservation, FileReadObservation

@pytest.fixture
def agent():
    """Create a test instance of AIFMDComplianceAgent."""
    return AIFMDComplianceAgent()

@pytest.fixture
def mock_state():
    """Create a mock State object."""
    state = Mock(spec=State)
    state.get_last_observation.return_value = None
    state.get_last_user_message.return_value = "Test message"
    return state

@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    return [
        PortfolioPosition(
            asset_id="EQUITY1",
            asset_type="equity",
            quantity=1000,
            market_value=100000.0,
            currency="EUR",
            sector="Technology",
            geography="Europe",
            rating=None,
            leverage=None,
            counterparty=None
        ),
        PortfolioPosition(
            asset_id="BOND1",
            asset_type="bond",
            quantity=500,
            market_value=50000.0,
            currency="EUR",
            sector="Finance",
            geography="Europe",
            rating="AAA",
            leverage=None,
            counterparty="Bank A"
        )
    ]

@pytest.fixture
def sample_returns():
    """Create sample return data for testing."""
    np.random.seed(42)
    return np.random.normal(0.0001, 0.02, 252)  # One year of daily returns

class TestAIFMDComplianceAgent:
    """Test suite for AIFMDComplianceAgent."""

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.name == "AIFMDComplianceAgent"
        assert agent.conversation_history == []
        assert agent.current_context is None
        assert agent.last_action is None
        assert isinstance(agent.portfolio_manager, PortfolioRiskManager)

    def test_can_handle(self, agent):
        """Test task handling capability detection."""
        assert agent.can_handle("Need AIFMD compliance review")
        assert agent.can_handle("Check portfolio risk management")
        assert agent.can_handle("Review Annex IV reporting")
        assert not agent.can_handle("Write a Python script")
        assert not agent.can_handle("Order lunch")

    def test_process_regulatory_query(self, agent):
        """Test regulatory query processing."""
        action, context = agent._process_regulatory_query("Show me AIFMD article 24")
        assert isinstance(action, BrowseURLAction)
        assert "eur-lex.europa.eu" in action.url
        
        action, context = agent._process_regulatory_query("General question")
        assert action is None
        assert context is None

    def test_process_reporting_query(self, agent):
        """Test reporting query processing."""
        action, context = agent._process_reporting_query("Help with Annex IV reporting")
        assert isinstance(action, BrowseURLAction)
        assert "esma.europa.eu" in action.url
        
        action, context = agent._process_reporting_query("Show technical guidance")
        assert isinstance(action, BrowseURLAction)
        assert "technical" in action.url.lower()

    def test_process_portfolio_data(self, agent, sample_portfolio, sample_returns):
        """Test portfolio data processing."""
        data = {
            "positions": [pos.__dict__ for pos in sample_portfolio],
            "returns": sample_returns.tolist(),
            "benchmark_returns": sample_returns.tolist(),
            "market_returns": sample_returns.tolist()
        }
        
        agent._process_portfolio_data(data)
        assert len(agent.current_portfolio) == 2
        assert isinstance(agent.historical_returns, np.ndarray)
        assert len(agent.historical_returns) == 252

    def test_process_portfolio_analysis(self, agent, sample_portfolio, sample_returns):
        """Test portfolio analysis processing."""
        # Setup portfolio data
        agent.current_portfolio = sample_portfolio
        agent.historical_returns = sample_returns
        
        # Test risk report generation
        action = agent._process_portfolio_analysis("Generate risk report")
        assert isinstance(action, FileWriteAction)
        assert "risk_report.md" in action.path
        
        # Test concentration analysis
        action = agent._process_portfolio_analysis("Show concentration risk")
        assert isinstance(action, MessageAction)
        assert "Concentration Analysis" in action.message
        
        # Test with no portfolio data
        agent.current_portfolio = None
        action = agent._process_portfolio_analysis("Generate risk report")
        assert isinstance(action, MessageAction)
        assert "No portfolio data available" in action.message

    def test_format_risk_report(self, agent):
        """Test risk report formatting."""
        report = {
            "risk_metrics": RiskMetrics(
                var_95=0.02,
                var_99=0.03,
                expected_shortfall=0.025,
                volatility=0.15,
                sharpe_ratio=1.2,
                tracking_error=0.03,
                information_ratio=0.8,
                beta=1.1,
                alpha=0.02
            ),
            "concentration_risks": {
                "asset_type": {"equity": 0.6, "bond": 0.4},
                "sector": {"Technology": 0.3, "Finance": 0.7}
            },
            "leverage_metrics": {
                "gross_leverage": 1.2,
                "net_leverage": 1.0
            },
            "liquidity_assessment": {
                "liquidity_profile": {
                    "highly_liquid": 0.7,
                    "moderately_liquid": 0.3
                },
                "liquidity_coverage_ratio": 1.5,
                "days_to_liquidate_90_percent": 3.0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        formatted = agent._format_risk_report(report)
        assert "Portfolio Risk Report" in formatted
        assert "Risk Metrics" in formatted
        assert "Concentration Analysis" in formatted
        assert "Leverage Analysis" in formatted
        assert "Liquidity Analysis" in formatted

    @patch('openhands.utils.llm.format_messages')
    def test_step_with_portfolio_request(self, mock_format_messages, agent, mock_state, sample_portfolio):
        """Test step method with portfolio-related request."""
        # Setup mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Portfolio analysis completed"
        agent.llm = Mock()
        agent.llm.completion.return_value = mock_response
        
        # Setup portfolio data
        agent.current_portfolio = sample_portfolio
        
        # Test with portfolio analysis request
        mock_state.get_last_user_message.return_value = "Generate risk report"
        action = agent.step(mock_state)
        assert isinstance(action, FileWriteAction)
        assert "risk_report" in action.path

    @patch('openhands.utils.llm.format_messages')
    def test_step_with_regulatory_request(self, mock_format_messages, agent, mock_state):
        """Test step method with regulatory request."""
        mock_state.get_last_user_message.return_value = "Show AIFMD Article 24"
        action = agent.step(mock_state)
        assert isinstance(action, BrowseURLAction)
        assert "eur-lex.europa.eu" in action.url

    def test_step_with_invalid_input(self, agent, mock_state):
        """Test step method with invalid input."""
        mock_state.get_last_user_message.return_value = None
        action = agent.step(mock_state)
        assert isinstance(action, AgentRejectAction)