from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from datetime import datetime
from dataclasses import dataclass

@dataclass
class FactorExposure:
    """Factor exposure analysis results."""
    factor_name: str
    beta: float
    t_stat: float
    r_squared: float
    p_value: float

@dataclass
class AttributionResult:
    """Performance attribution analysis results."""
    total_return: float
    factor_contributions: Dict[str, float]
    specific_return: float
    interaction_effects: Dict[str, float]
    r_squared: float

class QuantitativeAnalyzer:
    """Analyzer for quantitative analysis of securities."""

    def __init__(self):
        self.returns_data: Optional[pd.DataFrame] = None
        self.factor_data: Optional[pd.DataFrame] = None
        self.risk_free_rate: Optional[float] = None

    def analyze_returns_distribution(
        self,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Analyze the statistical properties of returns."""
        analysis = {
            'mean': returns.mean() * 252,  # Annualized mean
            'std': returns.std() * np.sqrt(252),  # Annualized volatility
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'jarque_bera': stats.jarque_bera(returns)[0],
            'jarque_bera_pvalue': stats.jarque_bera(returns)[1],
            'is_normal': stats.jarque_bera(returns)[1] > 0.05
        }
        
        # Calculate various quantiles
        for q in [0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99]:
            analysis[f'quantile_{int(q*100)}'] = returns.quantile(q)
        
        return analysis

    def perform_factor_analysis(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        risk_free_rate: Optional[float] = None
    ) -> List[FactorExposure]:
        """Perform factor analysis using multiple risk factors."""
        self.returns_data = returns
        self.factor_data = factors
        self.risk_free_rate = risk_free_rate

        # Calculate excess returns if risk-free rate is provided
        if risk_free_rate is not None:
            excess_returns = returns - risk_free_rate/252
        else:
            excess_returns = returns

        factor_exposures = []
        
        # Analyze each factor individually
        for factor_name in factors.columns:
            factor_returns = factors[factor_name]
            
            # Perform regression
            X = factor_returns.values.reshape(-1, 1)
            y = excess_returns.values
            
            # Add constant for intercept
            X = np.column_stack([np.ones_like(X), X])
            
            # Calculate regression coefficients
            try:
                beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
                
                # Calculate statistics
                y_pred = X.dot(beta)
                residuals = y - y_pred
                n = len(y)
                k = X.shape[1] - 1  # number of predictors
                mse = np.sum(residuals**2) / (n - k - 1)
                var_beta = mse * np.linalg.inv(X.T.dot(X))
                
                # Calculate t-statistics
                t_stat = beta[1] / np.sqrt(var_beta[1,1])
                
                # Calculate R-squared
                r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
                
                # Calculate p-value
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))
                
                factor_exposures.append(FactorExposure(
                    factor_name=factor_name,
                    beta=beta[1],
                    t_stat=t_stat,
                    r_squared=r_squared,
                    p_value=p_value
                ))
                
            except np.linalg.LinAlgError:
                continue

        return factor_exposures

    def perform_pca_analysis(
        self,
        returns: pd.DataFrame,
        n_components: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Perform Principal Component Analysis on returns."""
        # Standardize returns
        standardized_returns = (returns - returns.mean()) / returns.std()
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(standardized_returns)
        
        # Create DataFrame of principal components
        pc_df = pd.DataFrame(
            principal_components,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=returns.index
        )
        
        # Create DataFrame of component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=returns.columns
        )
        
        # Get explained variance ratios
        explained_variance = pca.explained_variance_ratio_
        
        return pc_df, loadings, explained_variance

    def perform_performance_attribution(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        weights: Optional[pd.Series] = None
    ) -> AttributionResult:
        """Perform returns-based performance attribution."""
        if weights is None:
            weights = pd.Series(1/len(factors.columns), index=factors.columns)

        # Calculate total return
        total_return = returns.mean() * 252  # Annualized

        # Perform multiple regression
        X = factors.values
        y = returns.values
        
        # Add constant for intercept
        X = np.column_stack([np.ones_like(X[:,0]), X])
        
        try:
            # Calculate regression coefficients
            beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            
            # Calculate factor contributions
            factor_contributions = {}
            for i, factor_name in enumerate(factors.columns):
                factor_contribution = beta[i+1] * factors[factor_name].mean() * 252
                factor_contributions[factor_name] = factor_contribution
            
            # Calculate specific return (alpha)
            specific_return = beta[0] * 252
            
            # Calculate predicted returns
            y_pred = X.dot(beta)
            
            # Calculate R-squared
            r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            
            # Calculate interaction effects
            interaction_effects = {}
            for i, factor1 in enumerate(factors.columns):
                for j, factor2 in enumerate(factors.columns[i+1:], i+1):
                    interaction = (factors[factor1] * factors[factor2]).mean() * 252
                    interaction_effects[f"{factor1}_{factor2}"] = interaction
            
        except np.linalg.LinAlgError:
            return AttributionResult(
                total_return=total_return,
                factor_contributions={},
                specific_return=total_return,
                interaction_effects={},
                r_squared=0.0
            )

        return AttributionResult(
            total_return=total_return,
            factor_contributions=factor_contributions,
            specific_return=specific_return,
            interaction_effects=interaction_effects,
            r_squared=r_squared
        )

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """Calculate advanced risk metrics."""
        # Annualization factor
        ann_factor = np.sqrt(252)
        
        # Calculate excess returns
        excess_returns = returns - risk_free_rate/252
        
        metrics = {
            'annualized_return': returns.mean() * 252,
            'annualized_volatility': returns.std() * ann_factor,
            'sharpe_ratio': (excess_returns.mean() / returns.std()) * ann_factor,
            'sortino_ratio': (excess_returns.mean() / returns[returns < 0].std()) * ann_factor,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'expected_shortfall_95': returns[returns <= np.percentile(returns, 5)].mean()
        }
        
        if benchmark_returns is not None:
            metrics.update({
                'beta': self._calculate_beta(returns, benchmark_returns),
                'alpha': self._calculate_alpha(returns, benchmark_returns, risk_free_rate),
                'tracking_error': np.std(returns - benchmark_returns) * ann_factor,
                'information_ratio': self._calculate_information_ratio(returns, benchmark_returns),
                'capture_ratio_up': self._calculate_capture_ratio(returns, benchmark_returns, up=True),
                'capture_ratio_down': self._calculate_capture_ratio(returns, benchmark_returns, up=False)
            })
        
        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = cum_returns / running_max - 1
        return drawdowns.min()

    def _calculate_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate beta relative to benchmark."""
        covariance = np.cov(returns, benchmark_returns)[0,1]
        variance = np.var(benchmark_returns)
        return covariance / variance

    def _calculate_alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float
    ) -> float:
        """Calculate Jensen's alpha."""
        beta = self._calculate_beta(returns, benchmark_returns)
        excess_return = returns.mean() * 252 - risk_free_rate
        excess_market_return = benchmark_returns.mean() * 252 - risk_free_rate
        return excess_return - beta * excess_market_return

    def _calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate information ratio."""
        active_returns = returns - benchmark_returns
        return active_returns.mean() / active_returns.std() * np.sqrt(252)

    def _calculate_capture_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        up: bool = True
    ) -> float:
        """Calculate up/down capture ratio."""
        if up:
            mask = benchmark_returns > 0
        else:
            mask = benchmark_returns < 0
            
        if not any(mask):
            return 0.0
            
        return returns[mask].mean() / benchmark_returns[mask].mean()

    def generate_quantitative_analysis_summary(
        self,
        returns_analysis: Dict[str, float],
        factor_exposures: List[FactorExposure],
        attribution: AttributionResult
    ) -> str:
        """Generate a summary of the quantitative analysis."""
        summary = []
        
        # Returns distribution summary
        summary.append("Returns Distribution Analysis:")
        summary.append(f"- Annualized Return: {returns_analysis['annualized_return']:.2%}")
        summary.append(f"- Annualized Volatility: {returns_analysis['annualized_volatility']:.2%}")
        summary.append(f"- Sharpe Ratio: {returns_analysis['sharpe_ratio']:.2f}")
        summary.append(f"- Maximum Drawdown: {returns_analysis['max_drawdown']:.2%}")
        summary.append(f"- Skewness: {returns_analysis['skewness']:.2f}")
        summary.append(f"- Kurtosis: {returns_analysis['kurtosis']:.2f}")
        
        # Factor analysis summary
        summary.append("\nFactor Analysis:")
        for factor in factor_exposures:
            if factor.p_value < 0.05:  # Only show significant factors
                summary.append(
                    f"- {factor.factor_name}: Beta = {factor.beta:.2f} "
                    f"(t-stat = {factor.t_stat:.2f}, R² = {factor.r_squared:.2%})"
                )
        
        # Performance attribution summary
        summary.append("\nPerformance Attribution:")
        summary.append(f"- Total Return: {attribution.total_return:.2%}")
        summary.append("- Factor Contributions:")
        for factor, contrib in attribution.factor_contributions.items():
            summary.append(f"  * {factor}: {contrib:.2%}")
        summary.append(f"- Specific Return: {attribution.specific_return:.2%}")
        summary.append(f"- Model R-squared: {attribution.r_squared:.2%}")
        
        return "\n".join(summary)