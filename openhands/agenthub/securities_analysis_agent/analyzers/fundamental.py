from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from ..models.security import SecurityFundamentals
from ..models.metrics import TimeFrame

class IndustryMetrics:
    """Industry-specific metrics and averages."""
    def __init__(self, industry_data: Dict[str, float]):
        self.avg_pe_ratio = industry_data.get('avg_pe_ratio')
        self.avg_pb_ratio = industry_data.get('avg_pb_ratio')
        self.avg_profit_margin = industry_data.get('avg_profit_margin')
        self.avg_roe = industry_data.get('avg_roe')
        self.avg_debt_to_equity = industry_data.get('avg_debt_to_equity')
        self.revenue_growth = industry_data.get('revenue_growth')
        self.market_size = industry_data.get('market_size')
        self.competition_level = industry_data.get('competition_level')

class FinancialStatements:
    """Container for financial statement data."""
    def __init__(
        self,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
        timeframe: TimeFrame = TimeFrame.QUARTERLY
    ):
        self.income_statement = income_statement
        self.balance_sheet = balance_sheet
        self.cash_flow = cash_flow
        self.timeframe = timeframe

class FundamentalAnalyzer:
    """Analyzer for fundamental analysis of securities."""

    def __init__(self):
        self.current_statements: Optional[FinancialStatements] = None
        self.industry_metrics: Optional[IndustryMetrics] = None

    def analyze_financial_statements(
        self, 
        statements: FinancialStatements
    ) -> Dict[str, float]:
        """Analyze financial statements and calculate key metrics."""
        self.current_statements = statements
        
        metrics = {
            'profitability': self._analyze_profitability(),
            'liquidity': self._analyze_liquidity(),
            'solvency': self._analyze_solvency(),
            'efficiency': self._analyze_efficiency(),
            'growth': self._analyze_growth_metrics(),
            'cash_flow': self._analyze_cash_flow()
        }
        
        return metrics

    def calculate_fundamentals(
        self,
        statements: FinancialStatements,
        market_price: float,
        shares_outstanding: float,
        industry_metrics: Optional[IndustryMetrics] = None
    ) -> SecurityFundamentals:
        """Calculate fundamental metrics for a security."""
        self.current_statements = statements
        self.industry_metrics = industry_metrics

        # Calculate market metrics
        market_cap = market_price * shares_outstanding
        
        # Get latest financial data
        latest_income = statements.income_statement.iloc[-1]
        latest_balance = statements.balance_sheet.iloc[-1]
        latest_cash_flow = statements.cash_flow.iloc[-1]
        
        # Calculate key metrics
        try:
            eps = latest_income['net_income'] / shares_outstanding
            pe_ratio = market_price / eps if eps != 0 else None
            
            book_value = latest_balance['total_assets'] - latest_balance['total_liabilities']
            pb_ratio = market_cap / book_value if book_value != 0 else None
            
            dividend_yield = (latest_income.get('dividends', 0) / shares_outstanding) / market_price
            
            debt = latest_balance.get('total_debt', 0)
            equity = latest_balance.get('total_equity', 0)
            debt_to_equity = debt / equity if equity != 0 else None
            
            current_assets = latest_balance.get('current_assets', 0)
            current_liabilities = latest_balance.get('current_liabilities', 0)
            current_ratio = current_assets / current_liabilities if current_liabilities != 0 else None
            
            quick_assets = current_assets - latest_balance.get('inventory', 0)
            quick_ratio = quick_assets / current_liabilities if current_liabilities != 0 else None
            
            roe = latest_income['net_income'] / equity if equity != 0 else None
            roa = latest_income['net_income'] / latest_balance['total_assets']
            
            revenue = latest_income['revenue']
            net_income = latest_income['net_income']
            ebitda = self._calculate_ebitda(latest_income)
            fcf = self._calculate_free_cash_flow(latest_cash_flow)
            
        except Exception as e:
            # Log error and return None for failed calculations
            print(f"Error calculating fundamentals: {str(e)}")
            return None
        
        return SecurityFundamentals(
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            dividend_yield=dividend_yield,
            debt_to_equity=debt_to_equity,
            current_ratio=current_ratio,
            quick_ratio=quick_ratio,
            roe=roe,
            roa=roa,
            eps=eps,
            revenue=revenue,
            net_income=net_income,
            ebitda=ebitda,
            fcf=fcf
        )

    def _analyze_profitability(self) -> Dict[str, float]:
        """Analyze profitability metrics."""
        if not self.current_statements:
            return {}
            
        latest = self.current_statements.income_statement.iloc[-1]
        revenue = latest.get('revenue', 0)
        
        metrics = {}
        if revenue != 0:
            metrics['gross_margin'] = latest.get('gross_profit', 0) / revenue
            metrics['operating_margin'] = latest.get('operating_income', 0) / revenue
            metrics['net_margin'] = latest.get('net_income', 0) / revenue
            
        return metrics

    def _analyze_liquidity(self) -> Dict[str, float]:
        """Analyze liquidity metrics."""
        if not self.current_statements:
            return {}
            
        latest = self.current_statements.balance_sheet.iloc[-1]
        current_liabilities = latest.get('current_liabilities', 0)
        
        metrics = {}
        if current_liabilities != 0:
            metrics['current_ratio'] = latest.get('current_assets', 0) / current_liabilities
            quick_assets = latest.get('current_assets', 0) - latest.get('inventory', 0)
            metrics['quick_ratio'] = quick_assets / current_liabilities
            
        return metrics

    def _analyze_solvency(self) -> Dict[str, float]:
        """Analyze solvency metrics."""
        if not self.current_statements:
            return {}
            
        latest = self.current_statements.balance_sheet.iloc[-1]
        equity = latest.get('total_equity', 0)
        
        metrics = {}
        if equity != 0:
            metrics['debt_to_equity'] = latest.get('total_debt', 0) / equity
            metrics['equity_multiplier'] = latest.get('total_assets', 0) / equity
            
        return metrics

    def _analyze_efficiency(self) -> Dict[str, float]:
        """Analyze efficiency metrics."""
        if not self.current_statements:
            return {}
            
        latest_income = self.current_statements.income_statement.iloc[-1]
        latest_balance = self.current_statements.balance_sheet.iloc[-1]
        
        metrics = {}
        revenue = latest_income.get('revenue', 0)
        if revenue != 0:
            metrics['asset_turnover'] = revenue / latest_balance.get('total_assets', 0)
            metrics['inventory_turnover'] = latest_income.get('cost_of_goods_sold', 0) / latest_balance.get('inventory', 1)
            
        return metrics

    def _analyze_growth_metrics(self) -> Dict[str, float]:
        """Analyze growth metrics."""
        if not self.current_statements:
            return {}
            
        df = self.current_statements.income_statement
        if len(df) < 2:
            return {}
            
        metrics = {}
        for column in ['revenue', 'net_income', 'operating_income']:
            if column in df.columns:
                current = df[column].iloc[-1]
                previous = df[column].iloc[-2]
                if previous != 0:
                    metrics[f'{column}_growth'] = (current - previous) / previous
                    
        return metrics

    def _analyze_cash_flow(self) -> Dict[str, float]:
        """Analyze cash flow metrics."""
        if not self.current_statements:
            return {}
            
        latest = self.current_statements.cash_flow.iloc[-1]
        
        metrics = {
            'operating_cash_flow': latest.get('operating_cash_flow', 0),
            'free_cash_flow': latest.get('free_cash_flow', 0),
            'capex': latest.get('capital_expenditures', 0)
        }
        
        return metrics

    def _calculate_ebitda(self, income_data: pd.Series) -> float:
        """Calculate EBITDA from income statement data."""
        net_income = income_data.get('net_income', 0)
        interest = income_data.get('interest_expense', 0)
        taxes = income_data.get('income_tax', 0)
        depreciation = income_data.get('depreciation', 0)
        amortization = income_data.get('amortization', 0)
        
        return net_income + interest + taxes + depreciation + amortization

    def _calculate_free_cash_flow(self, cash_flow_data: pd.Series) -> float:
        """Calculate free cash flow."""
        operating_cash_flow = cash_flow_data.get('operating_cash_flow', 0)
        capex = cash_flow_data.get('capital_expenditures', 0)
        
        return operating_cash_flow - capex

    def compare_to_industry(
        self,
        fundamentals: SecurityFundamentals,
        industry_metrics: IndustryMetrics
    ) -> Dict[str, Dict[str, float]]:
        """Compare security metrics to industry averages."""
        comparisons = {}
        
        # PE Ratio comparison
        if fundamentals.pe_ratio and industry_metrics.avg_pe_ratio:
            pe_diff = fundamentals.pe_ratio - industry_metrics.avg_pe_ratio
            comparisons['pe_ratio'] = {
                'security': fundamentals.pe_ratio,
                'industry': industry_metrics.avg_pe_ratio,
                'difference': pe_diff,
                'percent_diff': (pe_diff / industry_metrics.avg_pe_ratio) * 100
            }
            
        # ROE comparison
        if fundamentals.roe and industry_metrics.avg_roe:
            roe_diff = fundamentals.roe - industry_metrics.avg_roe
            comparisons['roe'] = {
                'security': fundamentals.roe,
                'industry': industry_metrics.avg_roe,
                'difference': roe_diff,
                'percent_diff': (roe_diff / industry_metrics.avg_roe) * 100
            }
            
        # Add more comparisons as needed
        
        return comparisons

    def generate_fundamental_analysis_summary(
        self,
        fundamentals: SecurityFundamentals,
        industry_comparison: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate a summary of the fundamental analysis."""
        summary = []
        
        # Valuation metrics
        summary.append("Valuation Metrics:")
        if fundamentals.pe_ratio:
            pe_analysis = industry_comparison.get('pe_ratio', {})
            if pe_analysis:
                summary.append(f"- P/E Ratio: {fundamentals.pe_ratio:.2f} "
                             f"(Industry: {pe_analysis['industry']:.2f}, "
                             f"Difference: {pe_analysis['percent_diff']:.1f}%)")
            else:
                summary.append(f"- P/E Ratio: {fundamentals.pe_ratio:.2f}")
                
        # Profitability
        summary.append("\nProfitability:")
        if fundamentals.roe:
            roe_analysis = industry_comparison.get('roe', {})
            if roe_analysis:
                summary.append(f"- Return on Equity: {fundamentals.roe:.2%} "
                             f"(Industry: {roe_analysis['industry']:.2%}, "
                             f"Difference: {roe_analysis['percent_diff']:.1f}%)")
            else:
                summary.append(f"- Return on Equity: {fundamentals.roe:.2%}")
                
        # Financial Health
        summary.append("\nFinancial Health:")
        if fundamentals.current_ratio:
            summary.append(f"- Current Ratio: {fundamentals.current_ratio:.2f}")
        if fundamentals.debt_to_equity:
            summary.append(f"- Debt to Equity: {fundamentals.debt_to_equity:.2f}")
            
        return "\n".join(summary)