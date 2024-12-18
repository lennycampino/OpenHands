from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PortfolioPosition:
    """Represents a position in the portfolio."""
    asset_id: str
    asset_type: str
    quantity: float
    market_value: float
    currency: str
    sector: Optional[str] = None
    geography: Optional[str] = None
    rating: Optional[str] = None
    leverage: Optional[float] = None
    counterparty: Optional[str] = None

@dataclass
class RiskMetrics:
    """Holds various risk metrics for a portfolio."""
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    expected_shortfall: float
    volatility: float
    sharpe_ratio: float
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None

class PortfolioRiskManager:
    """Advanced portfolio and risk management calculations."""

    def __init__(self):
        self.risk_free_rate = 0.02  # Configurable risk-free rate

    def calculate_portfolio_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation."""
        if len(returns) < 100:
            raise ValueError("Insufficient data for VaR calculation")
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(returns))
        return -sorted_returns[index]

    def calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var = self.calculate_portfolio_var(returns, confidence_level)
        return -np.mean(returns[returns <= -var])

    def calculate_portfolio_volatility(self, returns: np.ndarray) -> float:
        """Calculate portfolio volatility (annualized)."""
        return np.std(returns) * np.sqrt(252)  # Annualized assuming daily returns

    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe Ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def calculate_tracking_error(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate Tracking Error."""
        return np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)

    def calculate_information_ratio(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate Information Ratio."""
        active_returns = portfolio_returns - benchmark_returns
        return np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)

    def calculate_portfolio_beta(self, portfolio_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate portfolio beta."""
        covariance = np.cov(portfolio_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance

    def calculate_portfolio_alpha(self, portfolio_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate portfolio alpha."""
        beta = self.calculate_portfolio_beta(portfolio_returns, market_returns)
        portfolio_mean_return = np.mean(portfolio_returns) * 252
        market_mean_return = np.mean(market_returns) * 252
        return portfolio_mean_return - (self.risk_free_rate + beta * (market_mean_return - self.risk_free_rate))

    def calculate_concentration_risk(self, positions: List[PortfolioPosition]) -> Dict[str, Dict[str, float]]:
        """Calculate concentration risks across various dimensions."""
        total_value = sum(pos.market_value for pos in positions)
        
        concentrations = {
            'asset_type': {},
            'sector': {},
            'geography': {},
            'counterparty': {}
        }
        
        # Calculate concentrations by asset type
        for pos in positions:
            concentrations['asset_type'][pos.asset_type] = concentrations['asset_type'].get(
                pos.asset_type, 0) + pos.market_value / total_value
            
            if pos.sector:
                concentrations['sector'][pos.sector] = concentrations['sector'].get(
                    pos.sector, 0) + pos.market_value / total_value
            
            if pos.geography:
                concentrations['geography'][pos.geography] = concentrations['geography'].get(
                    pos.geography, 0) + pos.market_value / total_value
            
            if pos.counterparty:
                concentrations['counterparty'][pos.counterparty] = concentrations['counterparty'].get(
                    pos.counterparty, 0) + pos.market_value / total_value
        
        return concentrations

    def calculate_leverage_metrics(self, positions: List[PortfolioPosition]) -> Dict[str, float]:
        """Calculate various leverage metrics."""
        total_value = sum(pos.market_value for pos in positions)
        gross_exposure = sum(abs(pos.market_value) for pos in positions)
        net_exposure = sum(pos.market_value for pos in positions)
        
        return {
            'gross_leverage': gross_exposure / total_value,
            'net_leverage': net_exposure / total_value,
            'long_exposure': sum(pos.market_value for pos in positions if pos.market_value > 0) / total_value,
            'short_exposure': abs(sum(pos.market_value for pos in positions if pos.market_value < 0)) / total_value
        }

    def assess_liquidity_risk(self, positions: List[PortfolioPosition]) -> Dict[str, Union[float, Dict[str, float]]]:
        """Assess portfolio liquidity risk."""
        total_value = sum(pos.market_value for pos in positions)
        
        # Simplified liquidity buckets (should be enhanced with actual market data)
        liquidity_buckets = {
            'highly_liquid': 0.0,
            'moderately_liquid': 0.0,
            'less_liquid': 0.0,
            'illiquid': 0.0
        }
        
        # Example classification (should be enhanced with actual market data)
        for pos in positions:
            if pos.asset_type in ['cash', 'listed_equity', 'government_bonds']:
                liquidity_buckets['highly_liquid'] += pos.market_value
            elif pos.asset_type in ['corporate_bonds', 'etf']:
                liquidity_buckets['moderately_liquid'] += pos.market_value
            elif pos.asset_type in ['small_cap_equity', 'high_yield_bonds']:
                liquidity_buckets['less_liquid'] += pos.market_value
            else:
                liquidity_buckets['illiquid'] += pos.market_value
        
        # Convert to percentages
        liquidity_profile = {k: v/total_value for k, v in liquidity_buckets.items()}
        
        # Calculate liquidity coverage ratio (simplified)
        liquid_assets = liquidity_buckets['highly_liquid'] + 0.8 * liquidity_buckets['moderately_liquid']
        potential_outflows = total_value * 0.25  # Assuming 25% potential outflows in stress
        
        return {
            'liquidity_profile': liquidity_profile,
            'liquidity_coverage_ratio': liquid_assets / potential_outflows,
            'days_to_liquidate_90_percent': self._estimate_liquidation_time(liquidity_profile)
        }

    def _estimate_liquidation_time(self, liquidity_profile: Dict[str, float]) -> float:
        """Estimate days needed to liquidate 90% of the portfolio."""
        # Simplified estimation - should be enhanced with actual market data
        days_mapping = {
            'highly_liquid': 1,
            'moderately_liquid': 3,
            'less_liquid': 7,
            'illiquid': 30
        }
        
        total_days = sum(profile * days_mapping[bucket] for bucket, profile in liquidity_profile.items())
        return total_days

    def generate_risk_report(self, positions: List[PortfolioPosition], 
                           returns: np.ndarray,
                           benchmark_returns: Optional[np.ndarray] = None,
                           market_returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        risk_metrics = RiskMetrics(
            var_95=self.calculate_portfolio_var(returns, 0.95),
            var_99=self.calculate_portfolio_var(returns, 0.99),
            expected_shortfall=self.calculate_expected_shortfall(returns),
            volatility=self.calculate_portfolio_volatility(returns),
            sharpe_ratio=self.calculate_sharpe_ratio(returns)
        )
        
        if benchmark_returns is not None:
            risk_metrics.tracking_error = self.calculate_tracking_error(returns, benchmark_returns)
            risk_metrics.information_ratio = self.calculate_information_ratio(returns, benchmark_returns)
        
        if market_returns is not None:
            risk_metrics.beta = self.calculate_portfolio_beta(returns, market_returns)
            risk_metrics.alpha = self.calculate_portfolio_alpha(returns, market_returns)
        
        concentration_risks = self.calculate_concentration_risk(positions)
        leverage_metrics = self.calculate_leverage_metrics(positions)
        liquidity_assessment = self.assess_liquidity_risk(positions)
        
        return {
            'risk_metrics': risk_metrics,
            'concentration_risks': concentration_risks,
            'leverage_metrics': leverage_metrics,
            'liquidity_assessment': liquidity_assessment,
            'timestamp': datetime.now().isoformat()
        }