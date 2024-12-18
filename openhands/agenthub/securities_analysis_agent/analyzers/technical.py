from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
from enum import Enum

from ..models.metrics import TechnicalIndicators

class TrendDirection(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

class SignalStrength(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class TechnicalAnalyzer:
    """Analyzer for technical analysis of securities."""

    def __init__(self):
        self.price_data: Optional[pd.DataFrame] = None
        self.volume_data: Optional[pd.Series] = None
        self.indicators: Dict[str, pd.Series] = {}

    def analyze(
        self,
        prices: pd.DataFrame,
        volumes: pd.Series,
        timeframe: str = "daily"
    ) -> TechnicalIndicators:
        """Perform comprehensive technical analysis."""
        self.price_data = prices
        self.volume_data = volumes

        # Calculate all indicators
        self._calculate_moving_averages()
        self._calculate_momentum_indicators()
        self._calculate_volatility_indicators()
        self._calculate_volume_indicators()
        self._identify_support_resistance()

        return self._compile_technical_indicators()

    def _calculate_moving_averages(self) -> None:
        """Calculate various moving averages."""
        periods = [5, 10, 20, 50, 200]
        close_prices = self.price_data['close']

        # Simple Moving Averages
        for period in periods:
            self.indicators[f'sma_{period}'] = close_prices.rolling(window=period).mean()

        # Exponential Moving Averages
        for period in periods:
            self.indicators[f'ema_{period}'] = close_prices.ewm(span=period, adjust=False).mean()

        # MACD
        self.indicators['macd_line'] = close_prices.ewm(span=12, adjust=False).mean() - \
                                     close_prices.ewm(span=26, adjust=False).mean()
        self.indicators['macd_signal'] = self.indicators['macd_line'].ewm(span=9, adjust=False).mean()
        self.indicators['macd_histogram'] = self.indicators['macd_line'] - self.indicators['macd_signal']

    def _calculate_momentum_indicators(self) -> None:
        """Calculate momentum indicators."""
        close_prices = self.price_data['close']
        high_prices = self.price_data['high']
        low_prices = self.price_data['low']

        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.indicators['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low_14 = low_prices.rolling(window=14).min()
        high_14 = high_prices.rolling(window=14).max()
        self.indicators['stoch_k'] = 100 * (close_prices - low_14) / (high_14 - low_14)
        self.indicators['stoch_d'] = self.indicators['stoch_k'].rolling(window=3).mean()

        # Rate of Change
        self.indicators['roc'] = close_prices.pct_change(periods=12) * 100

    def _calculate_volatility_indicators(self) -> None:
        """Calculate volatility indicators."""
        high_prices = self.price_data['high']
        low_prices = self.price_data['low']
        close_prices = self.price_data['close']

        # Bollinger Bands
        sma_20 = close_prices.rolling(window=20).mean()
        std_20 = close_prices.rolling(window=20).std()
        self.indicators['bb_upper'] = sma_20 + (std_20 * 2)
        self.indicators['bb_lower'] = sma_20 - (std_20 * 2)
        self.indicators['bb_width'] = (self.indicators['bb_upper'] - self.indicators['bb_lower']) / sma_20

        # Average True Range (ATR)
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift())
        tr3 = abs(low_prices - close_prices.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.indicators['atr'] = tr.rolling(window=14).mean()

    def _calculate_volume_indicators(self) -> None:
        """Calculate volume-based indicators."""
        close_prices = self.price_data['close']
        volume = self.volume_data

        # On-Balance Volume (OBV)
        obv = pd.Series(index=close_prices.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        for i in range(1, len(close_prices)):
            if close_prices.iloc[i] > close_prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close_prices.iloc[i] < close_prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        self.indicators['obv'] = obv

        # Volume Moving Average
        self.indicators['volume_sma'] = volume.rolling(window=20).mean()

        # Price-Volume Trend
        self.indicators['pvt'] = volume * (close_prices.pct_change())

    def _identify_support_resistance(self) -> None:
        """Identify support and resistance levels."""
        def is_support(df: pd.DataFrame, i: int) -> bool:
            """Check if the point is a support level."""
            if i - 2 < 0 or i + 2 >= len(df):
                return False
            return (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                   df['low'].iloc[i] < df['low'].iloc[i+1] and
                   df['low'].iloc[i+1] < df['low'].iloc[i+2] and
                   df['low'].iloc[i-1] < df['low'].iloc[i-2])

        def is_resistance(df: pd.DataFrame, i: int) -> bool:
            """Check if the point is a resistance level."""
            if i - 2 < 0 or i + 2 >= len(df):
                return False
            return (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                   df['high'].iloc[i] > df['high'].iloc[i+1] and
                   df['high'].iloc[i+1] > df['high'].iloc[i+2] and
                   df['high'].iloc[i-1] > df['high'].iloc[i-2])

        levels = []
        for i in range(2, len(self.price_data) - 2):
            if is_support(self.price_data, i):
                levels.append((i, self.price_data['low'].iloc[i], 'support'))
            elif is_resistance(self.price_data, i):
                levels.append((i, self.price_data['high'].iloc[i], 'resistance'))

        self.support_levels = [level[1] for level in levels if level[2] == 'support']
        self.resistance_levels = [level[1] for level in levels if level[2] == 'resistance']

    def _determine_trend(self) -> TrendDirection:
        """Determine the overall trend direction."""
        close_prices = self.price_data['close']
        sma_20 = self.indicators['sma_20']
        sma_50 = self.indicators['sma_50']
        sma_200 = self.indicators['sma_200']

        # Get latest values
        current_price = close_prices.iloc[-1]
        current_sma20 = sma_20.iloc[-1]
        current_sma50 = sma_50.iloc[-1]
        current_sma200 = sma_200.iloc[-1]

        # Calculate price momentum
        price_momentum = (current_price - close_prices.iloc[-20]) / close_prices.iloc[-20]

        # Determine trend based on moving average alignment and momentum
        if (current_price > current_sma20 > current_sma50 > current_sma200 and
            price_momentum > 0.05):
            return TrendDirection.STRONG_UPTREND
        elif (current_price > current_sma20 > current_sma50 and
              price_momentum > 0):
            return TrendDirection.UPTREND
        elif (current_price < current_sma20 < current_sma50 < current_sma200 and
              price_momentum < -0.05):
            return TrendDirection.STRONG_DOWNTREND
        elif (current_price < current_sma20 < current_sma50 and
              price_momentum < 0):
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.SIDEWAYS

    def _calculate_trend_strength(self) -> float:
        """Calculate the strength of the current trend."""
        # Use ADX if available, otherwise calculate a simplified trend strength
        close_prices = self.price_data['close']
        sma_20 = self.indicators['sma_20']
        
        # Calculate price distance from moving average
        price_distance = abs(close_prices - sma_20) / sma_20
        
        # Calculate average distance over last 20 periods
        trend_strength = price_distance.rolling(window=20).mean().iloc[-1]
        
        # Normalize to 0-1 range
        return min(1.0, trend_strength * 5)

    def _compile_technical_indicators(self) -> TechnicalIndicators:
        """Compile all technical indicators into a single object."""
        trend_direction = self._determine_trend()
        trend_strength = self._calculate_trend_strength()

        return TechnicalIndicators(
            timestamp=datetime.now(),
            moving_averages={
                'sma_20': self.indicators['sma_20'].iloc[-1],
                'sma_50': self.indicators['sma_50'].iloc[-1],
                'sma_200': self.indicators['sma_200'].iloc[-1],
                'ema_20': self.indicators['ema_20'].iloc[-1]
            },
            rsi=self.indicators['rsi'].iloc[-1],
            macd={
                'line': self.indicators['macd_line'].iloc[-1],
                'signal': self.indicators['macd_signal'].iloc[-1],
                'histogram': self.indicators['macd_histogram'].iloc[-1]
            },
            bollinger_bands={
                'upper': self.indicators['bb_upper'].iloc[-1],
                'lower': self.indicators['bb_lower'].iloc[-1],
                'width': self.indicators['bb_width'].iloc[-1]
            },
            atr=self.indicators['atr'].iloc[-1],
            volume_metrics={
                'obv': self.indicators['obv'].iloc[-1],
                'volume_sma': self.indicators['volume_sma'].iloc[-1],
                'pvt': self.indicators['pvt'].iloc[-1]
            },
            momentum_indicators={
                'rsi': self.indicators['rsi'].iloc[-1],
                'stoch_k': self.indicators['stoch_k'].iloc[-1],
                'stoch_d': self.indicators['stoch_d'].iloc[-1],
                'roc': self.indicators['roc'].iloc[-1]
            },
            support_levels=self.support_levels[-3:] if self.support_levels else [],
            resistance_levels=self.resistance_levels[-3:] if self.resistance_levels else [],
            trend_strength=trend_strength,
            volatility_indicators={
                'atr': self.indicators['atr'].iloc[-1],
                'bb_width': self.indicators['bb_width'].iloc[-1]
            }
        )

    def generate_technical_analysis_summary(self) -> str:
        """Generate a summary of the technical analysis."""
        trend = self._determine_trend()
        indicators = self._compile_technical_indicators()
        
        summary = []
        summary.append(f"Technical Analysis Summary ({datetime.now().strftime('%Y-%m-%d')})")
        summary.append(f"\nTrend Analysis:")
        summary.append(f"- Direction: {trend.value}")
        summary.append(f"- Strength: {indicators.trend_strength:.2%}")
        
        summary.append("\nKey Indicators:")
        summary.append(f"- RSI: {indicators.momentum_indicators['rsi']:.2f}")
        summary.append(f"- MACD: {indicators.macd['histogram']:.2f}")
        summary.append(f"- ATR: {indicators.atr:.2f}")
        
        summary.append("\nSupport & Resistance:")
        if indicators.support_levels:
            summary.append(f"- Support Levels: {', '.join([f'${x:.2f}' for x in indicators.support_levels])}")
        if indicators.resistance_levels:
            summary.append(f"- Resistance Levels: {', '.join([f'${x:.2f}' for x in indicators.resistance_levels])}")
        
        summary.append("\nVolume Analysis:")
        summary.append(f"- OBV Trend: {'Positive' if indicators.volume_metrics['obv'] > 0 else 'Negative'}")
        
        return "\n".join(summary)