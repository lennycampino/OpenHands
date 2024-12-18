from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
import logging

from ..models.security import (
    SecurityPosition,
    SecurityPrice,
    SecurityIdentifiers,
    SecurityType,
    AssetClass
)

@dataclass
class DataValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missing_fields: List[str]
    data_quality_score: float

class MarketDataProcessor:
    """Processor for market data fetching and preprocessing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cached_data: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}

    def fetch_market_data(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch market data for a security."""
        try:
            # Check cache first
            cache_key = f"{ticker}_{interval}"
            if (cache_key in self.cached_data and
                cache_key in self.last_update and
                datetime.now() - self.last_update[cache_key] < timedelta(minutes=15)):
                return self.cached_data[cache_key]

            # Fetch data from yfinance
            security = yf.Ticker(ticker)
            data = security.history(
                start=start_date,
                end=end_date,
                interval=interval
            )

            # Validate and preprocess
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")

            # Clean and format data
            data = self._preprocess_market_data(data)

            # Update cache
            self.cached_data[cache_key] = data
            self.last_update[cache_key] = datetime.now()

            return data

        except Exception as e:
            self.logger.error(f"Error fetching market data for {ticker}: {str(e)}")
            raise

    def _preprocess_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data."""
        # Handle missing values
        data = data.fillna(method='ffill')
        
        # Calculate additional fields
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Add volume metrics
        data['volume_ma'] = data['Volume'].rolling(window=20).mean()
        data['relative_volume'] = data['Volume'] / data['volume_ma']
        
        return data

    def validate_data(self, data: pd.DataFrame) -> DataValidationResult:
        """Validate market data quality."""
        errors = []
        warnings = []
        missing_fields = []
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                missing_fields.append(col)
                errors.append(f"Missing required column: {col}")

        # Check for missing values
        missing_values = data.isnull().sum()
        for col, count in missing_values.items():
            if count > 0:
                warnings.append(f"Column {col} has {count} missing values")

        # Check for data continuity
        date_gaps = self._check_date_continuity(data.index)
        if date_gaps:
            warnings.append(f"Found {len(date_gaps)} gaps in time series")

        # Calculate data quality score
        total_points = 100
        # Reduce penalty for time series gaps as they're common in market data
        time_series_penalty = min(10, len(date_gaps) * 0.1) if date_gaps else 0
        
        # Calculate missing data penalty
        missing_data_penalty = sum(count/len(data) * 15 for col, count in missing_values.items() if count > 0)
        
        # Calculate other penalties
        error_penalty = len(errors) * 20
        warning_penalty = len([w for w in warnings if 'gaps in time series' not in w]) * 10
        
        # Calculate total deductions
        deductions = error_penalty + warning_penalty + missing_data_penalty + time_series_penalty
        
        # Calculate final score
        data_quality_score = max(0, min(100, total_points - deductions)) / 100

        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            missing_fields=missing_fields,
            data_quality_score=data_quality_score
        )

    def _check_date_continuity(self, index: pd.DatetimeIndex) -> List[Tuple[datetime, datetime]]:
        """Check for gaps in time series data."""
        gaps = []
        for i in range(1, len(index)):
            expected_diff = pd.Timedelta(days=1)
            if index[i] - index[i-1] > expected_diff:
                gaps.append((index[i-1], index[i]))
        return gaps

class PortfolioDataProcessor:
    """Processor for portfolio data."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.market_data_processor = MarketDataProcessor()

    def process_portfolio_data(
        self,
        positions_data: List[Dict],
        validate: bool = True
    ) -> Tuple[List[SecurityPosition], DataValidationResult]:
        """Process and validate portfolio data."""
        positions = []
        validation_results = []

        for pos_data in positions_data:
            try:
                # Create SecurityPosition object
                position = self._create_security_position(pos_data)
                
                # Fetch latest market data
                market_data = self.market_data_processor.fetch_market_data(
                    position.identifiers.ticker,
                    start_date=(datetime.now() - timedelta(days=5))
                )
                
                # Update position with latest market data
                position = self._update_position_with_market_data(position, market_data)
                
                positions.append(position)
                
                # Validate data if required
                if validate:
                    validation_result = self.market_data_processor.validate_data(market_data)
                    validation_results.append(validation_result)
                
            except Exception as e:
                self.logger.error(f"Error processing position: {str(e)}")
                continue

        # Aggregate validation results
        overall_validation = self._aggregate_validation_results(validation_results)
        
        return positions, overall_validation

    def _create_security_position(self, pos_data: Dict) -> SecurityPosition:
        """Create a SecurityPosition object from raw data."""
        return SecurityPosition(
            identifiers=SecurityIdentifiers(
                ticker=pos_data['ticker'],
                isin=pos_data.get('isin'),
                cusip=pos_data.get('cusip'),
                sedol=pos_data.get('sedol'),
                figi=pos_data.get('figi')
            ),
            security_type=SecurityType(pos_data['security_type']),
            asset_class=AssetClass(pos_data['asset_class']),
            quantity=float(pos_data['quantity']),
            cost_basis=float(pos_data['cost_basis']),
            currency=pos_data['currency'],
            current_price=float(pos_data.get('current_price', 0)),
            market_value=float(pos_data.get('market_value', 0)),
            unrealized_pnl=float(pos_data.get('unrealized_pnl', 0)),
            realized_pnl=float(pos_data.get('realized_pnl', 0)),
            position_date=datetime.fromisoformat(pos_data.get('position_date', datetime.now().isoformat()))
        )

    def _update_position_with_market_data(
        self,
        position: SecurityPosition,
        market_data: pd.DataFrame
    ) -> SecurityPosition:
        """Update position with latest market data."""
        if market_data.empty:
            return position

        latest_price = market_data['Close'].iloc[-1]
        position.current_price = latest_price
        position.market_value = position.quantity * latest_price
        position.unrealized_pnl = position.market_value - (position.quantity * position.cost_basis)
        
        return position

    def _aggregate_validation_results(
        self,
        results: List[DataValidationResult]
    ) -> DataValidationResult:
        """Aggregate multiple validation results."""
        if not results:
            return DataValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                missing_fields=[],
                data_quality_score=1.0
            )

        all_errors = []
        all_warnings = []
        all_missing_fields = []
        avg_quality_score = np.mean([r.data_quality_score for r in results])

        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_missing_fields.extend(result.missing_fields)

        return DataValidationResult(
            is_valid=len(all_errors) == 0,
            errors=list(set(all_errors)),
            warnings=list(set(all_warnings)),
            missing_fields=list(set(all_missing_fields)),
            data_quality_score=avg_quality_score
        )

    def get_historical_returns(
        self,
        positions: List[SecurityPosition],
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """Get historical returns for all positions."""
        returns_data = {}
        start_date = datetime.now() - timedelta(days=lookback_days)

        for position in positions:
            try:
                market_data = self.market_data_processor.fetch_market_data(
                    position.identifiers.ticker,
                    start_date=start_date
                )
                returns_data[position.identifiers.ticker] = market_data['returns']
            except Exception as e:
                self.logger.error(f"Error fetching returns for {position.identifiers.ticker}: {str(e)}")
                continue

        return pd.DataFrame(returns_data)