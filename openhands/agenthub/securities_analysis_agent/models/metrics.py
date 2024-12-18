from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from enum import Enum

class TimeFrame(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"

@dataclass
class PerformanceMetrics:
    """Performance metrics for a security or portfolio."""
    timeframe: TimeFrame
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    alpha: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float
    tracking_error: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    volatility: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    @classmethod
    def calculate(cls, returns: np.ndarray, benchmark_returns: np.ndarray, 
                 risk_free_rate: float, timeframe: TimeFrame) -> 'PerformanceMetrics':
        """Calculate performance metrics from return series."""
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Drawdown analysis
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Market metrics
        beta = np.cov(returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
        alpha = annualized_return - risk_free_rate - beta * (np.mean(benchmark_returns) * 252 - risk_free_rate)
        
        # Additional metrics
        tracking_error = np.std(returns - benchmark_returns) * np.sqrt(252)
        information_ratio = (np.mean(returns - benchmark_returns) * 252) / (tracking_error)
        
        # Win rate calculation
        positive_returns = np.sum(returns > 0)
        win_rate = positive_returns / len(returns)
        
        # Profit factor
        profit_factor = abs(np.sum(returns[returns > 0]) / np.sum(returns[returns < 0]))
        
        # Sortino ratio (using 0 as minimum acceptable return)
        downside_returns = returns[returns < 0]
        sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown)
        
        # Higher moments
        skewness = float(scipy.stats.skew(returns)) if scipy else None
        kurtosis = float(scipy.stats.kurtosis(returns)) if scipy else None
        
        return cls(
            timeframe=timeframe,
            start_date=datetime.now(),  # Should be from returns index
            end_date=datetime.now(),    # Should be from returns index
            total_return=total_return,
            annualized_return=annualized_return,
            alpha=alpha,
            beta=beta,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis
        )

@dataclass
class RiskMetrics:
    """Risk metrics for a security or portfolio."""
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    cvar_95: float  # 95% Conditional Value at Risk
    cvar_99: float  # 99% Conditional Value at Risk
    expected_shortfall: float
    volatility: float
    downside_volatility: float
    correlation_matrix: Optional[np.ndarray] = None
    stress_test_results: Optional[Dict[str, float]] = None
    scenario_analysis: Optional[Dict[str, float]] = None
    liquidity_score: Optional[float] = None
    concentration_score: Optional[float] = None

    @classmethod
    def calculate(cls, returns: np.ndarray, positions: List['SecurityPosition'] = None) -> 'RiskMetrics':
        """Calculate risk metrics from return series."""
        # Basic volatility
        volatility = np.std(returns) * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252)
        
        # VaR calculations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # CVaR/Expected Shortfall calculations
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])
        expected_shortfall = np.mean(returns[returns <= var_95])
        
        # Optional metrics if positions are provided
        liquidity_score = None
        concentration_score = None
        if positions:
            # Simple liquidity score based on position sizes
            total_value = sum(p.market_value for p in positions)
            position_weights = [p.market_value/total_value for p in positions]
            concentration_score = float(np.sum(np.square(position_weights)))  # Herfindahl-Hirschman Index
        
        return cls(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_shortfall=expected_shortfall,
            volatility=volatility,
            downside_volatility=downside_volatility,
            liquidity_score=liquidity_score,
            concentration_score=concentration_score
        )

@dataclass
class TechnicalIndicators:
    """Technical indicators for a security."""
    timestamp: datetime
    moving_averages: Dict[str, float]  # Different MA periods
    rsi: float
    macd: Dict[str, float]
    bollinger_bands: Dict[str, float]
    atr: float
    volume_metrics: Dict[str, float]
    momentum_indicators: Dict[str, float]
    support_levels: List[float]
    resistance_levels: List[float]
    trend_strength: float
    volatility_indicators: Dict[str, float]

    @classmethod
    def calculate(cls, prices: np.ndarray, volumes: np.ndarray) -> 'TechnicalIndicators':
        """Calculate technical indicators from price and volume data."""
        # Implementation will use TA-Lib or similar library
        pass