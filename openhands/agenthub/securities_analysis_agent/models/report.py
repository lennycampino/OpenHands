from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum
import json

class ReportType(Enum):
    SECURITY_ANALYSIS = "security_analysis"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    MARKET_ANALYSIS = "market_analysis"
    RISK_ANALYSIS = "risk_analysis"
    DUE_DILIGENCE = "due_diligence"

class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"

class Recommendation(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class ChartData:
    """Data for generating charts in reports."""
    chart_type: str
    title: str
    x_label: str
    y_label: str
    data: Dict[str, List[Union[float, str, datetime]]]
    annotations: Optional[List[Dict[str, Union[str, float, datetime]]]] = None

@dataclass
class ExecutiveSummary:
    """Executive summary section of a report."""
    key_findings: List[str]
    recommendations: List[str]
    risk_summary: str
    opportunity_summary: str
    confidence_level: ConfidenceLevel
    time_horizon: str

@dataclass
class AnalysisSection:
    """A section in the analysis report."""
    title: str
    content: str
    charts: Optional[List[ChartData]] = None
    tables: Optional[List[Dict[str, Union[str, float]]]] = None
    subsections: Optional[List['AnalysisSection']] = None

@dataclass
class AnalysisReport:
    """Complete analysis report."""
    report_id: str
    report_type: ReportType
    timestamp: datetime
    title: str
    executive_summary: ExecutiveSummary
    sections: List[AnalysisSection]
    recommendation: Optional[Recommendation] = None
    target_price: Optional[float] = None
    risk_rating: Optional[int] = None  # 1-5 scale
    analyst_name: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict:
        """Convert report to dictionary format."""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "executive_summary": {
                "key_findings": self.executive_summary.key_findings,
                "recommendations": self.executive_summary.recommendations,
                "risk_summary": self.executive_summary.risk_summary,
                "opportunity_summary": self.executive_summary.opportunity_summary,
                "confidence_level": self.executive_summary.confidence_level.value,
                "time_horizon": self.executive_summary.time_horizon
            },
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "charts": [chart.__dict__ for chart in section.charts] if section.charts else None,
                    "tables": section.tables,
                    "subsections": [subsec.__dict__ for subsec in section.subsections] if section.subsections else None
                }
                for section in self.sections
            ],
            "recommendation": self.recommendation.value if self.recommendation else None,
            "target_price": self.target_price,
            "risk_rating": self.risk_rating,
            "analyst_name": self.analyst_name,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class ReportTemplate:
    """Template for generating reports."""
    template_id: str
    template_type: ReportType
    sections: List[str]
    required_charts: List[str]
    required_metrics: List[str]
    style_guide: Dict[str, str]
    header_template: str
    footer_template: str
    page_layout: Dict[str, Union[int, str]]
    fonts: Dict[str, str]
    colors: Dict[str, str]

@dataclass
class ReportGenerationConfig:
    """Configuration for report generation."""
    template: ReportTemplate
    output_format: str  # 'pdf', 'html', 'docx'
    include_charts: bool = True
    include_tables: bool = True
    include_executive_summary: bool = True
    custom_branding: Optional[Dict[str, str]] = None
    language: str = "en"
    chart_style: str = "default"
    page_size: str = "A4"
    orientation: str = "portrait"
    compliance_check: bool = True