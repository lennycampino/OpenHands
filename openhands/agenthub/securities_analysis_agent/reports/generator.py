from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus.flowables import KeepTogether

from ..models.report import (
    AnalysisReport,
    ChartData,
    ReportGenerationConfig,
    ReportTemplate,
    ReportType
)
from ..models.security import SecurityAnalysis
from ..models.metrics import RiskMetrics, PerformanceMetrics

class ChartGenerator:
    """Generator for financial charts and visualizations."""

    @staticmethod
    def create_price_chart(
        price_data: pd.DataFrame,
        title: str,
        include_volume: bool = True
    ) -> go.Figure:
        """Create an interactive price chart with technical indicators."""
        fig = make_subplots(
            rows=2 if include_volume else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3] if include_volume else [1]
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add moving averages
        for period in [20, 50, 200]:
            ma = price_data['Close'].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=ma,
                    name=f'MA{period}',
                    line=dict(width=1)
                ),
                row=1, col=1
            )

        # Add volume bars
        if include_volume:
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['Volume'],
                    name='Volume'
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            yaxis2_title='Volume' if include_volume else None,
            xaxis_rangeslider_visible=False,
            height=800 if include_volume else 600
        )

        return fig

class PDFReportGenerator:
    """Generator for PDF reports."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Create custom styles with unique names
        self.styles.add(
            ParagraphStyle(
                name='CustomHeading1',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
        )
        self.styles.add(
            ParagraphStyle(
                name='CustomHeading2',
                parent=self.styles['Heading2'],
                fontSize=18,
                spaceAfter=20
            )
        )
        self.styles.add(
            ParagraphStyle(
                name='CustomBody',
                parent=self.styles['Normal'],
                fontSize=12,
                spaceAfter=12
            )
        )
        
        # Store references to our custom styles
        self.heading1_style = self.styles['CustomHeading1']
        self.heading2_style = self.styles['CustomHeading2']
        self.body_style = self.styles['CustomBody']

    def generate_report(
        self,
        report: AnalysisReport,
        config: ReportGenerationConfig,
        output_path: str
    ) -> None:
        """Generate a PDF report."""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Build the story (content)
        story = []
        
        # Add title
        story.append(Paragraph(report.title, self.heading1_style))
        story.append(Spacer(1, 12))
        
        # Add executive summary
        story.extend(self._create_executive_summary(report))
        
        # Add main sections
        for section in report.sections:
            story.extend(self._create_section(section))
            
        # Add footer
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.body_style
        ))
        
        # Build the PDF
        doc.build(story)

    def _create_executive_summary(self, report: AnalysisReport) -> List:
        """Create the executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.heading2_style))
        elements.append(Spacer(1, 12))
        
        # Key findings
        elements.append(Paragraph("Key Findings:", self.heading2_style))
        for finding in report.executive_summary.key_findings:
            elements.append(Paragraph(f"• {finding}", self.body_style))
        elements.append(Spacer(1, 12))
        
        # Recommendations
        elements.append(Paragraph("Recommendations:", self.heading2_style))
        for rec in report.executive_summary.recommendations:
            elements.append(Paragraph(f"• {rec}", self.body_style))
        elements.append(Spacer(1, 12))
        
        # Risk summary
        elements.append(Paragraph("Risk Summary:", self.heading2_style))
        elements.append(Paragraph(report.executive_summary.risk_summary, self.body_style))
        
        # Opportunity summary
        elements.append(Paragraph("Opportunity Summary:", self.heading2_style))
        elements.append(Paragraph(report.executive_summary.opportunity_summary, self.body_style))
        
        # Confidence level and time horizon
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(
            f"Confidence Level: {report.executive_summary.confidence_level.value}",
            self.body_style
        ))
        elements.append(Paragraph(
            f"Time Horizon: {report.executive_summary.time_horizon}",
            self.body_style
        ))
        elements.append(Spacer(1, 12))
        
        return elements

    def _create_section(self, section: 'AnalysisSection') -> List:
        """Create a report section."""
        elements = []
        
        # Section title
        elements.append(Paragraph(section.title, self.heading2_style))
        elements.append(Spacer(1, 12))
        
        # Section content
        elements.append(Paragraph(section.content, self.body_style))
        elements.append(Spacer(1, 12))
        
        # Add charts if any
        if section.charts:
            for chart in section.charts:
                elements.extend(self._create_chart(chart))
        
        # Add tables if any
        if section.tables:
            for table in section.tables:
                elements.extend(self._create_table(table))
        
        # Add subsections if any
        if section.subsections:
            for subsection in section.subsections:
                elements.extend(self._create_section(subsection))
                
        elements.append(Spacer(1, 20))
        return elements

    def _create_chart(self, chart_data: ChartData) -> List:
        """Create a chart for the report."""
        elements = []
        
        # Create the chart using plotly
        fig = go.Figure()
        
        # Add traces based on chart type
        if chart_data.chart_type == 'line':
            for name, values in chart_data.data.items():
                fig.add_trace(go.Scatter(
                    x=values['x'],
                    y=values['y'],
                    name=name,
                    mode='lines'
                ))
        elif chart_data.chart_type == 'bar':
            for name, values in chart_data.data.items():
                fig.add_trace(go.Bar(
                    x=values['x'],
                    y=values['y'],
                    name=name
                ))
        
        # Update layout
        fig.update_layout(
            title=chart_data.title,
            xaxis_title=chart_data.x_label,
            yaxis_title=chart_data.y_label
        )
        
        # Save to BytesIO
        img_bytes = BytesIO()
        fig.write_image(img_bytes, format='png')
        img_bytes.seek(0)
        
        # Add to PDF
        img = Image(img_bytes, width=6*inch, height=4*inch)
        elements.append(img)
        elements.append(Spacer(1, 12))
        
        return elements

    def _create_table(self, table_data: Dict[str, Union[str, float]]) -> List:
        """Create a table for the report."""
        elements = []
        
        # Convert dictionary to list of lists
        data = [[k, str(v)] for k, v in table_data.items()]
        
        # Create table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        return elements