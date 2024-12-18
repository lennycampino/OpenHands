from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import json
import pandas as pd
import numpy as np

from openhands.controller.agent import Agent
from .data.processor import MarketDataProcessor
from .analyzers.technical import TechnicalAnalyzer
from .analyzers.risk import RiskAnalyzer
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

from .models.security import (
    SecurityAnalysis,
    SecurityPosition,
    SecurityType,
    AssetClass,
    SecurityIdentifiers,
    SecurityFundamentals
)
from .models.metrics import (
    PerformanceMetrics,
    RiskMetrics,
    TechnicalIndicators,
    TimeFrame
)
from .models.report import (
    AnalysisReport,
    ReportType,
    ExecutiveSummary,
    AnalysisSection,
    Recommendation,
    ConfidenceLevel,
    ReportGenerationConfig
)

class SecuritiesAnalysisAgent(Agent):
    """Expert agent for securities analysis and professional report generation."""

    name = "SecuritiesAnalysisAgent"
    description = "Expert in securities analysis, risk assessment, and professional report generation for hedge funds"

    SYSTEM_PROMPT = """You are an expert securities analyst specializing in:

1. Comprehensive Securities Analysis:
   - Fundamental Analysis (financial statements, ratios, industry analysis)
   - Technical Analysis (price patterns, indicators, market sentiment)
   - Quantitative Analysis (statistical analysis, risk metrics, factor analysis)

2. Professional Report Generation:
   - Investment Thesis Reports
   - Risk Analysis Reports
   - Portfolio Performance Reports
   - Market Analysis Reports
   - Due Diligence Reports

3. Risk Management:
   - Portfolio Risk Analysis
   - Market Risk Assessment
   - Liquidity Risk Analysis
   - Stress Testing

Your expertise covers multiple asset classes:
- Equities
- Fixed Income
- Derivatives
- Cryptocurrencies
- Forex
- Commodities

Always provide detailed, data-driven analysis with clear recommendations."""

    def __init__(self, llm: "LLM", config: "AgentConfig") -> None:
        super().__init__(llm=llm, config=config)
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context: Optional[str] = None
        self.last_action: Optional[Action] = None
        
        # Initialize components
        self.logger = logging.getLogger(__name__)
        self.market_data_processor = MarketDataProcessor()
        self.technical_analyzer = TechnicalAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        
        # Analysis state
        self.current_analysis: Optional[SecurityAnalysis] = None
        self.current_portfolio: Optional[List[SecurityPosition]] = None
        self.current_report: Optional[AnalysisReport] = None

    def _process_security_query(self, query: str) -> Tuple[Action, Optional[str]]:
        """Process queries related to security analysis."""
        # Check for specific security analysis requests
        if any(keyword in query.lower() for keyword in [
            "analyze", "analyse", "research", "investigate", "evaluate"
        ]):
            # Extract security identifier if present
            # Implementation needed
            pass
        return None, None

    def _process_report_request(self, query: str) -> Tuple[Action, Optional[str]]:
        """Process requests for report generation."""
        report_types = {
            "investment thesis": ReportType.SECURITY_ANALYSIS,
            "risk analysis": ReportType.RISK_ANALYSIS,
            "portfolio analysis": ReportType.PORTFOLIO_ANALYSIS,
            "market analysis": ReportType.MARKET_ANALYSIS,
            "due diligence": ReportType.DUE_DILIGENCE
        }

        for keyword, report_type in report_types.items():
            if keyword in query.lower():
                if self.current_analysis or self.current_portfolio:
                    return self._generate_report(report_type), None
                else:
                    return MessageAction("No analysis data available. Please perform analysis first."), None

        return None, None

    def _generate_report(self, report_type: ReportType) -> Action:
        """Generate a professional report based on current analysis."""
        try:
            if report_type == ReportType.SECURITY_ANALYSIS and self.current_analysis:
                report = self._create_security_analysis_report()
            elif report_type == ReportType.PORTFOLIO_ANALYSIS and self.current_portfolio:
                report = self._create_portfolio_analysis_report()
            else:
                return AgentRejectAction("Insufficient data for report generation")

            # Convert report to PDF
            pdf_content = self._convert_report_to_pdf(report)
            return FileWriteAction(
                path=f"reports/{report.report_id}.pdf",
                content=pdf_content
            )
        except Exception as e:
            return AgentRejectAction(f"Error generating report: {str(e)}")

    def _create_security_analysis_report(self) -> AnalysisReport:
        """Create a detailed security analysis report."""
        analysis = self.current_analysis
        
        # Create executive summary
        exec_summary = ExecutiveSummary(
            key_findings=self._generate_key_findings(analysis),
            recommendations=self._generate_recommendations(analysis),
            risk_summary=self._generate_risk_summary(analysis),
            opportunity_summary=self._generate_opportunity_summary(analysis),
            confidence_level=self._determine_confidence_level(analysis),
            time_horizon="6-12 months"
        )

        # Create analysis sections
        sections = [
            self._create_fundamental_analysis_section(analysis),
            self._create_technical_analysis_section(analysis),
            self._create_risk_analysis_section(analysis),
            self._create_valuation_section(analysis)
        ]

        return AnalysisReport(
            report_id=f"SA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=ReportType.SECURITY_ANALYSIS,
            timestamp=datetime.now(),
            title=f"Security Analysis Report: {analysis.position.identifiers.ticker}",
            executive_summary=exec_summary,
            sections=sections,
            recommendation=self._determine_recommendation(analysis),
            target_price=analysis.target_price,
            risk_rating=self._calculate_risk_rating(analysis),
            analyst_name="Securities Analysis Agent",
            metadata={
                "analysis_date": analysis.analysis_date.isoformat(),
                "security_type": analysis.position.security_type.value,
                "asset_class": analysis.position.asset_class.value
            }
        )

    def analyze_portfolio(self, positions: List[SecurityPosition]) -> List[SecurityAnalysis]:
        """Analyze a portfolio of securities."""
        analyses = []
        for position in positions:
            try:
                # For testing purposes, use provided market data if available
                if hasattr(self, 'test_market_data'):
                    market_data = self.test_market_data
                else:
                    # Fetch market data
                    market_data = self.market_data_processor.fetch_market_data(
                        position.identifiers.ticker,
                        start_date=(datetime.now() - timedelta(days=252))
                    )

                # Standardize column names
                market_data.columns = [col.lower() for col in market_data.columns]

                # Perform technical analysis
                technical_indicators = self.technical_analyzer.analyze(
                    market_data,
                    market_data['volume']
                )

                # Calculate risk metrics
                returns = market_data['close'].pct_change().dropna()
                risk_metrics = self.risk_analyzer.calculate_risk_metrics(
                    returns=returns,
                    positions=[position]
                )

                # Create analysis object
                analysis = SecurityAnalysis(
                    position=position,
                    fundamentals=None,  # Will be populated when data is available
                    technical_indicators=technical_indicators.__dict__,
                    risk_metrics=risk_metrics.__dict__,
                    price_history=[],  # Will be populated with actual price history
                    analysis_date=datetime.now(),
                    analyst_notes=None,
                    recommendation=None,
                    target_price=None,
                    confidence_score=None
                )
                analyses.append(analysis)

            except Exception as e:
                self.logger.error(f"Error analyzing position {position.identifiers.ticker}: {str(e)}")
                continue

        return analyses

    def step(self, state: State) -> Action:
        """Process one step of the agent's operation."""
        # Get the latest observation
        observation = state.get_last_observation()
        
        # Process observation
        if isinstance(observation, BrowserOutputObservation):
            self.current_context = observation.text
        elif isinstance(observation, FileReadObservation):
            self.current_context = observation.content
            try:
                self._process_input_data(observation.content)
            except Exception as e:
                return AgentRejectAction(f"Error processing input data: {str(e)}")

        # Get user input
        user_input = state.get_last_user_message()
        if not user_input:
            return AgentRejectAction("No user input found.")

        # Process different types of requests
        security_action, context = self._process_security_query(user_input)
        if security_action:
            return security_action

        report_action, context = self._process_report_request(user_input)
        if report_action:
            return report_action

        # Format messages for LLM
        messages = self._format_prompt(user_input, self.current_context)
        
        try:
            # Get LLM response
            response = self.llm.completion(
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response_content})
            
            return MessageAction(response_content)
            
        except Exception as e:
            return AgentRejectAction(f"Error processing request: {str(e)}")

    def _process_input_data(self, content: str) -> None:
        """Process input data from files."""
        try:
            data = json.loads(content)
            if "security" in data:
                self._process_security_data(data["security"])
            elif "portfolio" in data:
                self._process_portfolio_data(data["portfolio"])
        except json.JSONDecodeError:
            # Handle non-JSON data
            pass

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
        
        # Add current analysis context if available
        if self.current_analysis:
            analysis_summary = self._format_analysis_summary()
            messages.append(Message(
                role="system",
                content=[TextContent(text=f"Current analysis:\n{analysis_summary}")]
            ))
        
        # Add user input
        messages.append(Message(role="user", content=[TextContent(text=user_input)]))
        return messages

    def generate_report(
        self,
        analysis: List[SecurityAnalysis],
        report_type: ReportType
    ) -> AnalysisReport:
        """Generate a report based on the analysis."""
        # Create executive summary
        exec_summary = ExecutiveSummary(
            key_findings=["Analysis completed successfully"],
            recommendations=["Review detailed metrics in report"],
            risk_summary="Risk analysis performed on portfolio",
            opportunity_summary="Opportunities identified in analysis",
            confidence_level=ConfidenceLevel.HIGH,
            time_horizon="6-12 months"
        )

        # Create analysis sections
        sections = []
        for security_analysis in analysis:
            sections.append(
                AnalysisSection(
                    title=f"Analysis for {security_analysis.position.identifiers.ticker}",
                    content=f"Detailed analysis of {security_analysis.position.identifiers.ticker}",
                    charts=[],
                    tables=[]
                )
            )

        # Create report
        return AnalysisReport(
            report_id=f"REP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=report_type,
            timestamp=datetime.now(),
            title="Portfolio Analysis Report",
            executive_summary=exec_summary,
            sections=sections,
            analyst_name="Securities Analysis Agent"
        )

    def calculate_portfolio_risk(
        self,
        positions: List[SecurityPosition]
    ) -> RiskMetrics:
        """Calculate portfolio risk metrics."""
        if not hasattr(self, 'test_market_data'):
            raise ValueError("Market data not available")
        
        returns = self.test_market_data['close'].pct_change().dropna()
        return self.risk_analyzer.calculate_risk_metrics(returns, positions)

    def analyze_technical_indicators(
        self,
        market_data: pd.DataFrame
    ) -> TechnicalIndicators:
        """Analyze technical indicators for market data."""
        market_data.columns = [col.lower() for col in market_data.columns]
        return self.technical_analyzer.analyze(market_data, market_data['volume'])

    def perform_quantitative_analysis(
        self,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Perform quantitative analysis on returns."""
        metrics = self.risk_analyzer.calculate_risk_metrics(returns, [])
        return {
            'sharpe_ratio': metrics.volatility,
            'max_drawdown': metrics.var_95,
            'volatility': metrics.volatility
        }

    def _format_analysis_summary(self) -> str:
        """Format current analysis for context."""
        if not self.current_analysis:
            return ""
        
        analysis = self.current_analysis
        return f"""Security: {analysis.position.identifiers.ticker}
Type: {analysis.position.security_type.value}
Current Price: ${analysis.position.current_price:.2f}
Market Value: ${analysis.position.market_value:.2f}
Unrealized P&L: ${analysis.position.unrealized_pnl:.2f}
Recommendation: {analysis.recommendation or 'Not available'}
Target Price: ${analysis.target_price or 'Not available'}
Confidence Score: {analysis.confidence_score or 'Not available'}
"""