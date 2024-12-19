# Example Prompts for OpenHands Agents

## AIFMD Compliance Agent

### Regulatory Guidance
```
Can you explain the key requirements of AIFMD Article 24 regarding reporting obligations?
```

### Risk Management Review
```
Please review our risk management system against AIFMD requirements. Here's our current setup:
{
    "risk_management": {
        "risk_limits": {
            "market_risk": "VaR limit of 5%",
            "leverage": "Maximum 200%",
            "liquidity": "25% portfolio liquidatable within 5 days"
        },
        "monitoring_frequency": "daily",
        "stress_testing": "monthly",
        "risk_reporting": "weekly to risk committee"
    }
}
```

### Portfolio Compliance Check
```
Can you analyze this portfolio for AIFMD compliance:
{
    "portfolio": {
        "leverage_ratio": 1.8,
        "largest_position": 0.15,
        "illiquid_assets": 0.25,
        "asset_classes": {
            "equities": 0.4,
            "fixed_income": 0.3,
            "alternatives": 0.2,
            "cash": 0.1
        }
    }
}
```

### Reporting Template Request
```
Generate an Annex IV reporting template for our quarterly AIFMD filing, focusing on:
- Portfolio composition
- Risk measures
- Leverage calculation
- Liquidity profile
```

### Compliance Framework Setup
```
Help us set up a compliance framework for a new private equity AIF with:
- €500M AUM
- Focus on European mid-market companies
- 5-year investment horizon
- Maximum 20% co-investment
```

## Securities Analysis Agent

### Portfolio Analysis
```
Analyze this portfolio and generate a comprehensive report:
{
    "positions": [
        {
            "ticker": "AAPL",
            "quantity": 1000,
            "cost_basis": 150.0,
            "current_price": 170.0,
            "sector": "Technology"
        },
        {
            "ticker": "MSFT",
            "quantity": 500,
            "cost_basis": 250.0,
            "current_price": 280.0,
            "sector": "Technology"
        }
    ]
}
```

### Risk Analysis
```
Perform a risk analysis on my portfolio with these stress scenarios:
- Market crash (-20%)
- Interest rate spike (+200bps)
- Tech sector decline (-15%)
- Currency crisis (USD/EUR -10%)
```

### Technical Analysis
```
Generate a technical analysis report for AAPL including:
- Moving averages (20, 50, 200 day)
- RSI and MACD indicators
- Support and resistance levels
- Volume analysis
- Price patterns
```

### Investment Research
```
Create a detailed investment thesis for NVDA covering:
- Fundamental analysis
- Industry position
- Growth prospects
- Risk factors
- Valuation analysis
- Price targets
```

### Performance Attribution
```
Analyze the performance of this portfolio over the last quarter:
{
    "portfolio_returns": 0.085,
    "benchmark_returns": 0.065,
    "sector_allocation": {
        "Technology": 0.35,
        "Healthcare": 0.25,
        "Financials": 0.20,
        "Consumer": 0.20
    },
    "risk_free_rate": 0.02
}
```

## Tips for Using the Agents

1. **Be Specific**
   - Provide concrete numbers and data
   - Specify time periods
   - Include relevant context

2. **Structure Your Input**
   - Use JSON format for data
   - Break down complex requests
   - Specify output format preferences

3. **Request Actionable Output**
   - Ask for specific recommendations
   - Request clear next steps
   - Specify required metrics

4. **Iterative Analysis**
   - Start with high-level analysis
   - Drill down into specific areas
   - Ask follow-up questions

5. **Combine Agent Capabilities**
   - Use AIFMD Agent for regulatory compliance
   - Use Securities Agent for investment analysis
   - Cross-reference findings between agents

## Example Combined Workflow

1. Initial Portfolio Setup
```
Securities Agent: Analyze this portfolio for investment opportunities and risks.
AIFMD Agent: Check if this portfolio meets regulatory requirements.
```

2. Risk Management
```
Securities Agent: Generate risk metrics and stress test results.
AIFMD Agent: Verify if risk management procedures comply with AIFMD.
```

3. Reporting
```
Securities Agent: Create detailed performance and risk reports.
AIFMD Agent: Format the reports to meet regulatory requirements.
```

4. Ongoing Monitoring
```
Securities Agent: Monitor portfolio performance and risk metrics.
AIFMD Agent: Track compliance status and reporting deadlines.
```

Remember: The agents can handle complex, multi-part requests and can provide detailed, professional-grade analysis and reports. Always provide as much context and data as possible for the most accurate and useful results.