from typing import Any, Dict, List, Optional, Tuple

from openhands.controller.agent import Agent
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
from openhands.core.message import Message, TextContent

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .portfolio_risk_manager import PortfolioRiskManager, PortfolioPosition, RiskMetrics

class AIFMDComplianceAgent(Agent):
    """Senior Portfolio and Risk Manager Agent specializing in AIFMD compliance and advanced portfolio management."""

    name = "AIFMDComplianceAgent"
    description = "Senior expert in AIFMD compliance, sophisticated risk management, and advanced portfolio management for Alternative Investment Funds"
    
    # Base system prompt defining the agent's expertise and capabilities
    SYSTEM_PROMPT = """You are an expert compliance agent specializing in Alternative Investment Funds (AIFs) in Europe.
Your expertise covers:

1. AIFMD (Alternative Investment Fund Managers Directive):
   - Complete regulatory framework understanding
   - Implementation requirements
   - Ongoing compliance monitoring

2. Risk Management:
   - Risk management systems and procedures
   - Risk measurement methodologies
   - Stress testing requirements
   - Risk limits and monitoring

3. Portfolio Management:
   - Investment restrictions
   - Portfolio composition requirements
   - Leverage calculations
   - Valuation procedures

4. Reporting Requirements:
   - AIFMD reporting (Annex IV)
   - Risk reporting
   - Portfolio composition reporting
   - Regulatory disclosures

5. Fund Manager Processes:
   - Regulatory requirements
   - Compliance workflows
   - Documentation requirements
   - Operational procedures

You can:
- Interpret regulatory requirements
- Review compliance procedures
- Provide regulatory guidance
- Create compliance frameworks
- Assess risk management systems
- Generate compliance reports
- Evaluate portfolio compliance

Always provide accurate, regulation-based advice and clearly reference relevant AIFMD articles or guidelines when applicable."""

    def __init__(self, llm: "LLM", config: "AgentConfig") -> None:
        super().__init__(llm=llm, config=config)
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context: Optional[str] = None
        self.last_action: Optional[Action] = None
        self.portfolio_manager = PortfolioRiskManager()
        self.current_portfolio: Optional[List[PortfolioPosition]] = None
        self.historical_returns: Optional[np.ndarray] = None
        self.benchmark_returns: Optional[np.ndarray] = None
        self.market_returns: Optional[np.ndarray] = None

    def _format_prompt(self, user_input: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        """Format the prompt with conversation history and context."""
        messages = []
        
        # Add system prompt
        messages.append(Message(role="system", content=[TextContent(text=self.SYSTEM_PROMPT)]))
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append(Message(
                role=msg["role"],
                content=[TextContent(text=msg["content"])]
            ))
        
        # Add current context if available
        if context:
            messages.append(Message(
                role="system",
                content=[TextContent(text=f"Current context:\n{context}")]
            ))
        
        # Add user input
        messages.append(Message(role="user", content=[TextContent(text=user_input)]))
        return messages

    def _process_regulatory_query(self, query: str) -> Tuple[Action, Optional[str]]:
        """Process queries related to regulatory requirements."""
        # First, check if we need to fetch regulatory information
        if any(keyword in query.lower() for keyword in ["article", "regulation", "directive", "requirement"]):
            # Use the consolidated AIFMD text from EUR-Lex
            return BrowseURLAction(
                url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02011L0061-20210802"
            ), None
        return None, None

    def _process_reporting_query(self, query: str) -> Tuple[Action, Optional[str]]:
        """Process queries related to reporting requirements."""
        # Map of keywords to specific ESMA resources
        resource_map = {
            "reporting guidelines": "https://www.esma.europa.eu/sites/default/files/library/2015/11/2014-869.pdf",  # Main reporting guidelines
            "annex iv": "https://www.esma.europa.eu/sites/default/files/library/2015/11/2014-869.pdf",  # Annex IV reporting
            "disclosure": "https://www.esma.europa.eu/sites/default/files/library/2015/11/2014-869.pdf",  # Disclosure requirements
            "technical": "https://www.esma.europa.eu/sites/default/files/library/2013-1358_aifmd_reporting_it_technical_guidance-revision5.xlsx",  # Technical guidance
            "validation": "https://www.esma.europa.eu/document/aifmd-reporting-it-technical-guidance-rev-6-updated",  # Validation rules
            "q&a": "https://www.esma.europa.eu/sites/default/files/library/esma34-32-352_qa_aifmd.pdf"  # Q&A document
        }

        # Check for specific keywords and return appropriate resource
        for keyword, url in resource_map.items():
            if keyword in query.lower():
                return BrowseURLAction(url=url), None

        # Only return action if there's a specific match
        return None, None

    def step(self, state: State) -> Action:
        """Process one step of the agent's operation."""
        # Get the latest observation if available
        observation = state.get_last_observation()
        
        # If we have a browser output observation, process it
        if isinstance(observation, BrowserOutputObservation):
            self.current_context = observation.text
        elif isinstance(observation, FileReadObservation):
            self.current_context = observation.content
            # Try to parse portfolio data if available
            try:
                import json
                data = json.loads(self.current_context)
                if 'positions' in data:
                    self._process_portfolio_data(data)
            except:
                pass

        # Get user input from the last message
        user_input = state.get_last_user_message()
        if not user_input:
            return AgentRejectAction("No user input found.")

        # Check if we need to fetch regulatory or reporting information first
        regulatory_action, context = self._process_regulatory_query(user_input)
        if regulatory_action:
            self.last_action = regulatory_action
            return regulatory_action

        reporting_action, context = self._process_reporting_query(user_input)
        if reporting_action:
            self.last_action = reporting_action
            return reporting_action

        # Then check if this is a portfolio analysis request
        portfolio_action = self._process_portfolio_analysis(user_input)
        if portfolio_action:
            return portfolio_action

        # Format messages for LLM
        messages = self._format_prompt(user_input, self.current_context)
        
        # Add portfolio context if available
        if self.current_portfolio:
            portfolio_summary = {
                "num_positions": len(self.current_portfolio),
                "asset_types": list(set(pos.asset_type for pos in self.current_portfolio)),
                "total_value": sum(pos.market_value for pos in self.current_portfolio)
            }
            messages.append({
                "role": "system",
                "content": f"Current portfolio context:\n{json.dumps(portfolio_summary, indent=2)}"
            })
        
        # Get LLM response
        try:
            response = self.llm.completion(
                messages=messages,
                temperature=0.3,  # Lower temperature for more precise responses
                max_tokens=2000
            )
            
            # Extract the response content
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response_content})
            
            # Check if we need to create a document
            if any(keyword in user_input.lower() for keyword in ["create", "generate", "write", "prepare"]):
                if any(doc_type in user_input.lower() for doc_type in [
                    "policy", "procedure", "framework", "report",
                    "risk assessment", "portfolio analysis"
                ]):
                    # Generate comprehensive report if portfolio data is available
                    if self.current_portfolio and any(term in user_input.lower() for term in [
                        "risk", "portfolio", "analysis", "assessment"
                    ]):
                        report = self.portfolio_manager.generate_risk_report(
                            self.current_portfolio,
                            self.historical_returns,
                            self.benchmark_returns,
                            self.market_returns
                        )
                        return FileWriteAction(
                            path="portfolio_risk_report.md",
                            content=self._format_risk_report(report)
                        )
                    else:
                        return FileWriteAction(
                            path="compliance_document.md",
                            content=response_content
                        )
            
            # Return message action with the response
            return MessageAction(response_content)
            
        except Exception as e:
            return AgentRejectAction(f"Error processing request: {str(e)}")

    def _process_portfolio_analysis(self, query: str) -> Optional[Action]:
        """Process portfolio analysis requests."""
        if not self.current_portfolio:
            return MessageAction("No portfolio data available. Please provide portfolio data first.")
        
        try:
            if "risk report" in query.lower():
                report = self.portfolio_manager.generate_risk_report(
                    self.current_portfolio,
                    self.historical_returns,
                    self.benchmark_returns,
                    self.market_returns
                )
                return FileWriteAction(
                    path="risk_report.md",
                    content=self._format_risk_report(report)
                )
            
            if "concentration risk" in query.lower():
                risks = self.portfolio_manager.calculate_concentration_risk(self.current_portfolio)
                return MessageAction(self._format_concentration_risks(risks))
            
            if "leverage analysis" in query.lower():
                metrics = self.portfolio_manager.calculate_leverage_metrics(self.current_portfolio)
                return MessageAction(self._format_leverage_metrics(metrics))
            
            if "liquidity assessment" in query.lower():
                assessment = self.portfolio_manager.assess_liquidity_risk(self.current_portfolio)
                return MessageAction(self._format_liquidity_assessment(assessment))
            
        except Exception as e:
            return AgentRejectAction(f"Error in portfolio analysis: {str(e)}")
        
        return None

    def _format_risk_report(self, report: Dict[str, Any]) -> str:
        """Format risk report into markdown."""
        metrics = report['risk_metrics']
        
        md = "# Portfolio Risk Report\n\n"
        md += f"Generated on: {report['timestamp']}\n\n"
        
        md += "## Risk Metrics\n"
        md += f"* VaR (95%): {metrics.var_95:.2%}\n"
        md += f"* VaR (99%): {metrics.var_99:.2%}\n"
        md += f"* Expected Shortfall: {metrics.expected_shortfall:.2%}\n"
        md += f"* Volatility (annualized): {metrics.volatility:.2%}\n"
        md += f"* Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n"
        
        if metrics.tracking_error is not None:
            md += f"* Tracking Error: {metrics.tracking_error:.2%}\n"
            md += f"* Information Ratio: {metrics.information_ratio:.2f}\n"
        
        if metrics.beta is not None:
            md += f"* Beta: {metrics.beta:.2f}\n"
            md += f"* Alpha (annualized): {metrics.alpha:.2%}\n"
        
        md += "\n## Concentration Analysis\n"
        for category, risks in report['concentration_risks'].items():
            md += f"\n### {category.replace('_', ' ').title()}\n"
            for item, value in risks.items():
                md += f"* {item}: {value:.2%}\n"
        
        md += "\n## Leverage Analysis\n"
        for metric, value in report['leverage_metrics'].items():
            md += f"* {metric.replace('_', ' ').title()}: {value:.2f}x\n"
        
        md += "\n## Liquidity Analysis\n"
        liquidity = report['liquidity_assessment']
        md += "\n### Liquidity Profile\n"
        for bucket, percentage in liquidity['liquidity_profile'].items():
            md += f"* {bucket.replace('_', ' ').title()}: {percentage:.2%}\n"
        
        md += f"\nLiquidity Coverage Ratio: {liquidity['liquidity_coverage_ratio']:.2f}\n"
        md += f"Estimated Days to Liquidate 90%: {liquidity['days_to_liquidate_90_percent']:.1f}\n"
        
        return md

    def _format_concentration_risks(self, risks: Dict[str, Dict[str, float]]) -> str:
        """Format concentration risks into readable text."""
        output = "Portfolio Concentration Analysis:\n\n"
        
        for category, values in risks.items():
            output += f"{category.replace('_', ' ').title()}:\n"
            sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=True)
            for item, value in sorted_items:
                output += f"- {item}: {value:.2%}\n"
            output += "\n"
        
        return output

    def _format_leverage_metrics(self, metrics: Dict[str, float]) -> str:
        """Format leverage metrics into readable text."""
        output = "Portfolio Leverage Analysis:\n\n"
        
        for metric, value in metrics.items():
            output += f"- {metric.replace('_', ' ').title()}: {value:.2f}x\n"
        
        return output

    def _format_liquidity_assessment(self, assessment: Dict[str, Union[float, Dict[str, float]]]) -> str:
        """Format liquidity assessment into readable text."""
        output = "Portfolio Liquidity Assessment:\n\n"
        
        output += "Liquidity Profile:\n"
        for bucket, percentage in assessment['liquidity_profile'].items():
            output += f"- {bucket.replace('_', ' ').title()}: {percentage:.2%}\n"
        
        output += f"\nLiquidity Coverage Ratio: {assessment['liquidity_coverage_ratio']:.2f}\n"
        output += f"Estimated Days to Liquidate 90%: {assessment['days_to_liquidate_90_percent']:.1f}\n"
        
        return output

    def _process_portfolio_data(self, data: Dict[str, Any]) -> None:
        """Process and store portfolio data."""
        positions = []
        for pos_data in data.get('positions', []):
            positions.append(PortfolioPosition(**pos_data))
        self.current_portfolio = positions
        
        if 'returns' in data:
            self.historical_returns = np.array(data['returns'])
        if 'benchmark_returns' in data:
            self.benchmark_returns = np.array(data['benchmark_returns'])
        if 'market_returns' in data:
            self.market_returns = np.array(data['market_returns'])

    def can_handle(self, task_description: str) -> bool:
        """Determine if this agent can handle the given task."""
        keywords = [
            "aifmd", "aif", "alternative investment fund",
            "compliance", "regulation", "directive",
            "risk management", "portfolio management",
            "private equity", "reporting", "annex iv",
            "portfolio analysis", "risk metrics", "var",
            "leverage", "liquidity", "concentration",
            "stress test", "performance attribution"
        ]
        return any(keyword in task_description.lower() for keyword in keywords)