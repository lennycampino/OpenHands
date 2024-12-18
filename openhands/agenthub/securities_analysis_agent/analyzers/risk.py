from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from dataclasses import dataclass

from ..models.security import SecurityPosition
from ..models.metrics import RiskMetrics, PerformanceMetrics, TimeFrame

@dataclass
class StressTestScenario:
    """Definition of a stress test scenario."""
    name: str
    market_shock: float  # Market return shock
    volatility_shock: float  # Volatility increase factor
    correlation_shock: float  # Correlation adjustment
    liquidity_shock: float  # Liquidity reduction factor
    interest_rate_shock: float  # Interest rate change in basis points
    fx_shock: Optional[Dict[str, float]] = None  # Currency pair shocks

@dataclass
class StressTestResult:
    """Results of a stress test."""
    scenario: StressTestScenario
    portfolio_impact: float
    var_impact: float
    liquidity_impact: float
    position_impacts: Dict[str, float]
    risk_factor_contributions: Dict[str, float]

class RiskAnalyzer:
    """Analyzer for risk analysis and stress testing."""

    def __init__(self):
        self.historical_returns: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.current_positions: Optional[List[SecurityPosition]] = None

    def calculate_risk_metrics(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        positions: List[SecurityPosition],
        confidence_level: float = 0.95,
        time_horizon: int = 10
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        self.historical_returns = returns
        self.current_positions = positions
        
        # Handle both Series and DataFrame inputs
        if isinstance(returns, pd.Series):
            self.correlation_matrix = pd.DataFrame([[1.0]], columns=['returns'], index=['returns'])
        else:
            self.correlation_matrix = returns.corr()

        # Handle single security vs portfolio
        if isinstance(returns, pd.Series):
            portfolio_returns = returns
        else:
            # Calculate position weights
            total_value = sum(pos.market_value for pos in positions)
            weights = np.array([pos.market_value/total_value for pos in positions])
            portfolio_returns = returns.dot(weights)

        # Calculate VaR
        var_95 = self._calculate_var(portfolio_returns, 0.95)
        var_99 = self._calculate_var(portfolio_returns, 0.99)

        # Calculate CVaR/Expected Shortfall
        cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
        cvar_99 = self._calculate_cvar(portfolio_returns, 0.99)

        # Calculate volatility metrics
        volatility = portfolio_returns.std() * np.sqrt(252)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Calculate liquidity and concentration metrics
        liquidity_score = self._calculate_liquidity_score(positions)
        concentration_score = self._calculate_concentration_score(positions)

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_shortfall=cvar_95,
            volatility=volatility,
            downside_volatility=downside_volatility,
            correlation_matrix=self.correlation_matrix.values,
            liquidity_score=liquidity_score,
            concentration_score=concentration_score
        )

    def run_stress_tests(
        self,
        positions: List[SecurityPosition],
        returns: pd.DataFrame,
        scenarios: List[StressTestScenario]
    ) -> Dict[str, StressTestResult]:
        """Run multiple stress test scenarios."""
        results = {}
        for scenario in scenarios:
            result = self._run_single_stress_test(positions, returns, scenario)
            results[scenario.name] = result
        return results

    def _run_single_stress_test(
        self,
        positions: List[SecurityPosition],
        returns: pd.DataFrame,
        scenario: StressTestScenario
    ) -> StressTestResult:
        # Initialize correlation matrix if not already set
        if not hasattr(self, 'correlation_matrix') or self.correlation_matrix is None:
            self.correlation_matrix = returns.corr()
        """Run a single stress test scenario."""
        # Apply market shock
        shocked_returns = returns * (1 + scenario.market_shock)
        
        # Apply volatility shock
        shocked_vol = returns.std() * scenario.volatility_shock
        
        # Calculate position impacts
        position_impacts = {}
        total_value = sum(pos.market_value for pos in positions)
        
        for pos in positions:
            # Calculate position-specific impact
            beta = 1.0  # Should be calculated based on position characteristics
            position_impact = -(pos.market_value / total_value) * scenario.market_shock * beta
            position_impacts[pos.identifiers.ticker] = position_impact
        
        # Calculate portfolio-level impact
        portfolio_impact = sum(position_impacts.values())
        
        # Calculate risk metric impacts
        original_var = self._calculate_var(returns, 0.95)
        shocked_var = self._calculate_var(shocked_returns, 0.95)
        var_impact = (shocked_var - original_var) / original_var
        
        # Calculate liquidity impact
        liquidity_impact = self._calculate_liquidity_impact(positions, scenario.liquidity_shock)
        
        # Calculate risk factor contributions
        risk_factor_contributions = {
            'market_risk': scenario.market_shock * portfolio_impact,
            'volatility_risk': (scenario.volatility_shock - 1) * shocked_vol.mean(),
            'liquidity_risk': liquidity_impact,
            'correlation_risk': scenario.correlation_shock * self.correlation_matrix.mean().mean()
        }
        
        return StressTestResult(
            scenario=scenario,
            portfolio_impact=portfolio_impact,
            var_impact=var_impact,
            liquidity_impact=liquidity_impact,
            position_impacts=position_impacts,
            risk_factor_contributions=risk_factor_contributions
        )

    def _calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float,
        method: str = 'historical'
    ) -> float:
        """Calculate Value at Risk."""
        if method == 'historical':
            return -np.percentile(returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            z_score = stats.norm.ppf(confidence_level)
            return -(returns.mean() + z_score * returns.std())
        else:
            raise ValueError(f"Unsupported VaR method: {method}")

    def _calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = self._calculate_var(returns, confidence_level)
        return -returns[returns <= -var].mean()

    def _calculate_liquidity_score(
        self,
        positions: List[SecurityPosition]
    ) -> float:
        """Calculate portfolio liquidity score."""
        total_value = sum(pos.market_value for pos in positions)
        
        # Simple liquidity score based on position sizes
        # Should be enhanced with actual trading volume data
        liquidity_scores = []
        for pos in positions:
            position_weight = pos.market_value / total_value
            # Larger positions are considered less liquid
            liquidity_score = 1 - (position_weight ** 0.5)
            liquidity_scores.append(liquidity_score)
            
        return np.mean(liquidity_scores)

    def _calculate_concentration_score(
        self,
        positions: List[SecurityPosition]
    ) -> float:
        """Calculate portfolio concentration score using HHI."""
        total_value = sum(pos.market_value for pos in positions)
        weights = [pos.market_value/total_value for pos in positions]
        return sum(w*w for w in weights)

    def _calculate_liquidity_impact(
        self,
        positions: List[SecurityPosition],
        liquidity_shock: float
    ) -> float:
        """Calculate the impact of a liquidity shock."""
        total_value = sum(pos.market_value for pos in positions)
        
        # Calculate liquidation impact
        impact = 0
        for pos in positions:
            position_weight = pos.market_value / total_value
            # Larger positions have higher impact under stress
            position_impact = position_weight * liquidity_shock * (1 + position_weight)
            impact += position_impact
            
        return impact

    def generate_risk_analysis_summary(
        self,
        risk_metrics: RiskMetrics,
        stress_results: Optional[Dict[str, StressTestResult]] = None
    ) -> str:
        """Generate a summary of the risk analysis."""
        summary = []
        summary.append("Risk Analysis Summary")
        summary.append("\nKey Risk Metrics:")
        summary.append(f"- Value at Risk (95%): {risk_metrics.var_95:.2%}")
        summary.append(f"- Expected Shortfall (95%): {risk_metrics.cvar_95:.2%}")
        summary.append(f"- Portfolio Volatility: {risk_metrics.volatility:.2%}")
        summary.append(f"- Downside Volatility: {risk_metrics.downside_volatility:.2%}")
        
        if risk_metrics.liquidity_score is not None:
            summary.append(f"\nLiquidity Analysis:")
            summary.append(f"- Liquidity Score: {risk_metrics.liquidity_score:.2f}")
            
        if risk_metrics.concentration_score is not None:
            summary.append(f"\nConcentration Analysis:")
            summary.append(f"- Concentration Score (HHI): {risk_metrics.concentration_score:.2f}")
        
        if stress_results:
            summary.append("\nStress Test Results:")
            for scenario_name, result in stress_results.items():
                summary.append(f"\n{scenario_name}:")
                summary.append(f"- Portfolio Impact: {result.portfolio_impact:.2%}")
                summary.append(f"- VaR Impact: {result.var_impact:.2%}")
                summary.append(f"- Liquidity Impact: {result.liquidity_impact:.2%}")
        
        return "\n".join(summary)

    def get_default_stress_scenarios(self) -> List[StressTestScenario]:
        """Get default stress test scenarios."""
        return [
            StressTestScenario(
                name="Market Crash",
                market_shock=-0.20,
                volatility_shock=2.0,
                correlation_shock=0.3,
                liquidity_shock=0.5,
                interest_rate_shock=100
            ),
            StressTestScenario(
                name="Recession",
                market_shock=-0.15,
                volatility_shock=1.5,
                correlation_shock=0.2,
                liquidity_shock=0.3,
                interest_rate_shock=-50
            ),
            StressTestScenario(
                name="Interest Rate Spike",
                market_shock=-0.10,
                volatility_shock=1.3,
                correlation_shock=0.1,
                liquidity_shock=0.2,
                interest_rate_shock=200
            ),
            StressTestScenario(
                name="Currency Crisis",
                market_shock=-0.12,
                volatility_shock=1.4,
                correlation_shock=0.15,
                liquidity_shock=0.4,
                interest_rate_shock=150,
                fx_shock={"EUR/USD": -0.15, "GBP/USD": -0.12}
            )
        ]