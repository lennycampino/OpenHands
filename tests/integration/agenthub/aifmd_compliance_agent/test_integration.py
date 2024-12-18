import pytest
import json
import numpy as np
from datetime import datetime, timedelta

from openhands.agenthub.aifmd_compliance_agent.agent import AIFMDComplianceAgent
from openhands.agenthub.aifmd_compliance_agent.portfolio_risk_manager import PortfolioPosition
from openhands.controller.state import State
from openhands.events.action import (
    Action,
    AgentFinishAction,
    BrowseURLAction,
    FileWriteAction,
    MessageAction,
)
from openhands.events.observation import FileReadObservation, BrowserOutputObservation

@pytest.fixture
def agent():
    """Create a test instance of AIFMDComplianceAgent."""
    return AIFMDComplianceAgent()

@pytest.fixture
def sample_portfolio_data():
    """Create sample portfolio data for testing."""
    # Create a diversified portfolio
    positions = [
        PortfolioPosition(
            asset_id="EQUITY_EU_TECH_1",
            asset_type="equity",
            quantity=10000,
            market_value=1000000.0,
            currency="EUR",
            sector="Technology",
            geography="Europe",
            rating=None,
            leverage=None,
            counterparty=None
        ),
        PortfolioPosition(
            asset_id="BOND_EU_GOV_1",
            asset_type="government_bond",
            quantity=2000000,
            market_value=2000000.0,
            currency="EUR",
            sector="Government",
            geography="Europe",
            rating="AAA",
            leverage=None,
            counterparty="German Government"
        ),
        PortfolioPosition(
            asset_id="PE_FUND_1",
            asset_type="private_equity",
            quantity=1,
            market_value=500000.0,
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
            market_value=750000.0,
            currency="EUR",
            sector="Various",
            geography="Global",
            rating=None,
            leverage=2.0,
            counterparty="Hedge Fund B"
        )
    ]
    
    # Generate sample returns data
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, 252)  # Daily returns for 1 year
    benchmark_returns = np.random.normal(0.0001, 0.018, 252)  # Benchmark returns
    market_returns = np.random.normal(0.0001, 0.015, 252)  # Market returns
    
    return {
        "positions": [pos.__dict__ for pos in positions],
        "returns": returns.tolist(),
        "benchmark_returns": benchmark_returns.tolist(),
        "market_returns": market_returns.tolist()
    }

class TestAIFMDComplianceAgentIntegration:
    """Integration tests for AIFMDComplianceAgent."""

    def test_full_portfolio_analysis_workflow(self, agent, sample_portfolio_data):
        """Test complete portfolio analysis workflow."""
        # Create a mock state
        state = State()
        
        # Step 1: Load portfolio data
        portfolio_json = json.dumps(sample_portfolio_data)
        state.add_observation(FileReadObservation(content=portfolio_json))
        
        # Initial analysis request
        state.add_user_message("Analyze the portfolio and generate a comprehensive risk report")
        action = agent.step(state)
        
        # Verify risk report generation
        assert isinstance(action, FileWriteAction)
        assert "risk_report" in action.path
        report_content = action.content
        
        # Verify report sections
        assert "Portfolio Risk Report" in report_content
        assert "Risk Metrics" in report_content
        assert "Concentration Analysis" in report_content
        assert "Leverage Analysis" in report_content
        assert "Liquidity Analysis" in report_content
        
        # Step 2: Request specific risk analysis
        state.add_user_message("Show me the concentration risks in the portfolio")
        action = agent.step(state)
        
        # Verify concentration analysis
        assert isinstance(action, MessageAction)
        concentration_analysis = action.message
        assert "Concentration Analysis" in concentration_analysis
        assert "Technology" in concentration_analysis
        assert "Government" in concentration_analysis
        
        # Step 3: Check leverage analysis
        state.add_user_message("What are the current leverage metrics?")
        action = agent.step(state)
        
        # Verify leverage analysis
        assert isinstance(action, MessageAction)
        leverage_analysis = action.message
        assert "Leverage Analysis" in leverage_analysis
        
        # Step 4: Request liquidity assessment
        state.add_user_message("Assess the portfolio's liquidity risk")
        action = agent.step(state)
        
        # Verify liquidity assessment
        assert isinstance(action, MessageAction)
        liquidity_assessment = action.message
        assert "Liquidity Assessment" in liquidity_assessment

    def test_regulatory_compliance_workflow(self, agent):
        """Test regulatory compliance workflow."""
        state = State()
        
        # Step 1: Request AIFMD reporting guidance
        state.add_user_message("What are the reporting requirements under AIFMD Article 24?")
        action = agent.step(state)
        
        # Verify ESMA guidance access
        assert isinstance(action, BrowseURLAction)
        assert "esma.europa.eu" in action.url
        
        # Add mock response from ESMA website
        state.add_observation(BrowserOutputObservation(
            text="Article 24 reporting requirements include: frequency of reporting, "
                 "content requirements, and submission deadlines."
        ))
        
        # Step 2: Request technical guidance
        state.add_user_message("Show me the technical specifications for Annex IV reporting")
        action = agent.step(state)
        
        # Verify technical guidance access
        assert isinstance(action, BrowseURLAction)
        assert "technical" in action.url.lower()
        
        # Step 3: Request validation rules
        state.add_user_message("What are the current validation rules for AIFMD reporting?")
        action = agent.step(state)
        
        # Verify validation rules access
        assert isinstance(action, BrowseURLAction)
        assert "validation" in action.url.lower()

    def test_combined_portfolio_and_regulatory_workflow(self, agent, sample_portfolio_data):
        """Test combined portfolio analysis and regulatory compliance workflow."""
        state = State()
        
        # Load portfolio data
        portfolio_json = json.dumps(sample_portfolio_data)
        state.add_observation(FileReadObservation(content=portfolio_json))
        
        # Step 1: Request compliance check with portfolio context
        state.add_user_message(
            "Check if our portfolio complies with AIFMD requirements and generate a report"
        )
        action = agent.step(state)
        
        # Verify comprehensive report generation
        assert isinstance(action, FileWriteAction)
        report_content = action.content
        assert "Portfolio Risk Report" in report_content
        assert "Risk Metrics" in report_content
        
        # Step 2: Request specific regulatory guidance
        state.add_user_message(
            "What are the reporting requirements for our private equity positions?"
        )
        action = agent.step(state)
        
        # Verify regulatory guidance with portfolio context
        assert isinstance(action, MessageAction) or isinstance(action, BrowseURLAction)
        
        # Step 3: Request leverage compliance check
        state.add_user_message(
            "Are our current leverage levels compliant with AIFMD requirements?"
        )
        action = agent.step(state)
        
        # Verify leverage compliance analysis
        assert isinstance(action, MessageAction)
        leverage_analysis = action.message
        assert "Leverage" in leverage_analysis

    def test_error_handling_and_recovery(self, agent):
        """Test error handling and recovery capabilities."""
        state = State()
        
        # Test with invalid portfolio data
        state.add_observation(FileReadObservation(content="invalid json"))
        state.add_user_message("Analyze portfolio risk")
        action = agent.step(state)
        
        # Verify error handling
        assert isinstance(action, MessageAction)
        assert "No portfolio data available" in action.message
        
        # Test recovery with valid request
        state.add_user_message("Show me AIFMD reporting requirements")
        action = agent.step(state)
        
        # Verify recovery
        assert isinstance(action, BrowseURLAction)
        assert "esma.europa.eu" in action.url