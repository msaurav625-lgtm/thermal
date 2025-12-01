"""
Export and Report Generation for Nanofluid Simulator

This module provides functionality to export simulation results in various formats
and generate comprehensive PDF reports.
"""

import pandas as pd
from typing import List, Dict, Optional, TYPE_CHECKING
from datetime import datetime
import os

if TYPE_CHECKING:
    from reportlab.platypus import Table

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, Image, PageBreak, Frame, PageTemplate
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    # Create dummy classes for type hints
    Table = object

from .enhanced_simulator import EnhancedSimulationResult


class ResultExporter:
    """Export simulation results to various formats."""
    
    @staticmethod
    def to_csv(results: List[EnhancedSimulationResult], filename: str):
        """
        Export results to CSV file.
        
        Args:
            results: List of simulation results
            filename: Output CSV filename
        """
        data = [r.to_dict() for r in results]
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    
    @staticmethod
    def to_excel(
        results: List[EnhancedSimulationResult],
        filename: str,
        sheet_name: str = "Results"
    ):
        """
        Export results to Excel file with formatting.
        
        Args:
            results: List of simulation results
            filename: Output Excel filename
            sheet_name: Name of the Excel sheet
        """
        data = [r.to_dict() for r in results]
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Format headers
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
                cell.fill = cell.fill.copy(
                    patternType='solid',
                    fgColor='366092'
                )
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    @staticmethod
    def parametric_to_excel(
        results_dict: Dict[str, List[EnhancedSimulationResult]],
        filename: str,
        parameter_name: str = "Parameter"
    ):
        """
        Export parametric study results to Excel with multiple sheets.
        
        Args:
            results_dict: Dictionary mapping model names to result lists
            filename: Output Excel filename
            parameter_name: Name of the varying parameter
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Create summary sheet with all models
            summary_data = []
            
            for model_name, results in results_dict.items():
                for result in results:
                    data_dict = result.to_dict()
                    data_dict['model'] = model_name
                    summary_data.append(data_dict)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create individual sheets for each model
            for model_name, results in results_dict.items():
                data = [r.to_dict() for r in results]
                df = pd.DataFrame(data)
                
                # Truncate sheet name if too long
                sheet_name = model_name[:31]  # Excel limit is 31 characters
                df.to_excel(writer, sheet_name=sheet_name, index=False)


class ReportGenerator:
    """Generate comprehensive PDF reports."""
    
    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF generation. "
                "Install with: pip install reportlab"
            )
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2196F3'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1976D2'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#424242'),
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
    
    def generate_report(
        self,
        results: List[EnhancedSimulationResult],
        config: Dict,
        filename: str,
        include_plots: Optional[List[str]] = None
    ):
        """
        Generate a comprehensive PDF report.
        
        Args:
            results: List of simulation results
            config: Configuration dictionary
            filename: Output PDF filename
            include_plots: List of plot image filenames to include
        """
        doc = SimpleDocTemplate(filename, pagesize=A4)
        story = []
        
        # Title
        title = Paragraph(
            "Nanofluid Thermal Conductivity<br/>Simulation Report",
            self.styles['CustomTitle']
        )
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Date and time
        date_str = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        date_para = Paragraph(
            f"<i>Generated: {date_str}</i>",
            self.styles['Normal']
        )
        story.append(date_para)
        story.append(Spacer(1, 0.5*inch))
        
        # Configuration section
        story.append(Paragraph("Configuration", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        config_text = self._format_configuration(config)
        for line in config_text:
            story.append(Paragraph(line, self.styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Results section
        story.append(Paragraph("Simulation Results", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        # Results table
        results_table = self._create_results_table(results)
        story.append(results_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detailed results
        story.append(Paragraph("Detailed Analysis", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        for result in results:
            model_para = Paragraph(
                f"<b>{result.model_name}</b>",
                self.styles['SubSection']
            )
            story.append(model_para)
            
            details = self._format_result_details(result)
            for detail in details:
                story.append(Paragraph(detail, self.styles['Normal']))
                story.append(Spacer(1, 0.05*inch))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Include plots if provided
        if include_plots:
            story.append(PageBreak())
            story.append(Paragraph("Visualizations", self.styles['SectionHeader']))
            story.append(Spacer(1, 0.2*inch))
            
            for plot_file in include_plots:
                if os.path.exists(plot_file):
                    img = Image(plot_file, width=5*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3*inch))
        
        # Build PDF
        doc.build(story)
    
    def _format_configuration(self, config: Dict) -> List[str]:
        """Format configuration dictionary for report."""
        lines = []
        
        if 'base_fluid' in config and config['base_fluid']:
            bf = config['base_fluid']
            lines.append(f"<b>Base Fluid:</b> {bf.get('name', 'N/A')}")
            lines.append(
                f"<b>Thermal Conductivity:</b> "
                f"{bf.get('thermal_conductivity', 0):.4f} W/m·K"
            )
            lines.append(
                f"<b>Density:</b> {bf.get('density', 0):.2f} kg/m³"
            )
            lines.append(
                f"<b>Viscosity:</b> {bf.get('viscosity', 0):.6f} Pa·s"
            )
        
        if 'temperature_C' in config:
            lines.append(f"<b>Temperature:</b> {config['temperature_C']:.1f} °C")
        
        if 'nanoparticles' in config and config['nanoparticles']:
            lines.append(f"<b>Nanoparticles:</b>")
            for i, np in enumerate(config['nanoparticles'], 1):
                lines.append(
                    f"  {i}. {np.get('name', 'N/A')} ({np.get('formula', '')}): "
                    f"φ = {np.get('volume_fraction_percent', 0):.2f}%"
                )
        
        if config.get('is_hybrid'):
            lines.append("<b>Type:</b> <i>Hybrid Nanofluid</i>")
        else:
            lines.append("<b>Type:</b> <i>Mono Nanofluid</i>")
        
        return lines
    
    def _create_results_table(self, results: List[EnhancedSimulationResult]):
        """Create formatted results table."""
        # Table data
        data = [
            ['Model', 'k_eff (W/m·K)', 'Enhancement (%)', 'Viscosity (Pa·s)', 'Pr']
        ]
        
        for result in results:
            row = [
                result.model_name,
                f"{result.k_effective:.6f}",
                f"{result.enhancement_k:.2f}" if result.enhancement_k else "N/A",
                f"{result.mu_effective:.6f}" if result.mu_effective else "N/A",
                f"{result.pr_effective:.4f}" if result.pr_effective else "N/A"
            ]
            data.append(row)
        
        # Create table
        table = Table(data, colWidths=[2.2*inch, 1.3*inch, 1.2*inch, 1.3*inch, 0.8*inch])
        
        # Style table
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196F3')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Body styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        return table
    
    def _format_result_details(self, result: EnhancedSimulationResult) -> List[str]:
        """Format detailed result information."""
        lines = []
        
        lines.append(
            f"Thermal Conductivity: <b>{result.k_effective:.6f} W/m·K</b>"
        )
        
        if result.enhancement_k is not None:
            lines.append(
                f"Enhancement: <b>{result.enhancement_k:.2f}%</b>"
            )
        
        if result.mu_effective is not None:
            lines.append(
                f"Dynamic Viscosity: {result.mu_effective:.6f} Pa·s "
                f"(+{result.enhancement_mu:.2f}%)"
            )
        
        if result.rho_effective is not None:
            lines.append(f"Density: {result.rho_effective:.2f} kg/m³")
        
        if result.cp_effective is not None:
            lines.append(f"Specific Heat: {result.cp_effective:.2f} J/kg·K")
        
        if result.alpha_effective is not None:
            lines.append(
                f"Thermal Diffusivity: {result.alpha_effective*1e7:.4f} × 10⁻⁷ m²/s"
            )
        
        if result.pr_effective is not None:
            lines.append(f"Prandtl Number: {result.pr_effective:.4f}")
        
        return lines


def export_results(
    results: List[EnhancedSimulationResult],
    filename: str,
    format: str = 'csv'
):
    """
    Quick export function.
    
    Args:
        results: List of simulation results
        filename: Output filename (with extension)
        format: Export format ('csv', 'excel', 'xlsx')
    """
    exporter = ResultExporter()
    
    format_lower = format.lower()
    
    if format_lower == 'csv':
        exporter.to_csv(results, filename)
    elif format_lower in ['excel', 'xlsx']:
        exporter.to_excel(results, filename)
    else:
        raise ValueError(f"Unsupported format: {format}")


def generate_pdf_report(
    results: List[EnhancedSimulationResult],
    config: Dict,
    filename: str,
    plots: Optional[List[str]] = None
):
    """
    Quick PDF report generation function.
    
    Args:
        results: List of simulation results
        config: Configuration dictionary
        filename: Output PDF filename
        plots: Optional list of plot image files to include
    """
    generator = ReportGenerator()
    generator.generate_report(results, config, filename, plots)
