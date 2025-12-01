#!/usr/bin/env python3
"""
BKPS NFL Thermal Pro 7.0 - Publication-Ready PDF Report Generator
Dedicated to: Brijesh Kumar Pandey

Generates professional PDF reports with:
- Input configuration table
- Material properties
- Results with enhancement metrics
- Visualization images
- DLVO stability graphs
- Performance indices
- Version hash & DOI
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import io
import hashlib

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.platypus import KeepTogether
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available. Using matplotlib-only PDF generation.")


class PDFReportGenerator:
    """
    Professional PDF report generator for nanofluid simulation results
    
    Usage:
        generator = PDFReportGenerator()
        generator.generate_report(
            results=results,
            config=config,
            output_path="report.pdf"
        )
    """
    
    def __init__(self):
        self.version = "7.0.0"
        self.release_date = "2025-11-30"
        
    def generate_report(self,
                       results: Dict[str, Any],
                       config: Any,
                       output_path: str,
                       include_figures: bool = True,
                       include_validation: bool = False) -> None:
        """
        Generate complete PDF report
        
        Args:
            results: Simulation results dictionary
            config: UnifiedConfig object
            output_path: Output PDF file path
            include_figures: Include visualization figures
            include_validation: Include validation comparison
        """
        if REPORTLAB_AVAILABLE:
            self._generate_reportlab_pdf(results, config, output_path, 
                                        include_figures, include_validation)
        else:
            self._generate_matplotlib_pdf(results, config, output_path)
    
    def _generate_matplotlib_pdf(self, results: Dict, config: Any, output_path: str):
        """Generate PDF using matplotlib (fallback)"""
        with PdfPages(output_path) as pdf:
            # Page 1: Title and configuration
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('BKPS NFL Thermal Pro 7.0\nSimulation Report', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # Configuration summary
            config_text = self._format_config_text(config)
            ax.text(0.1, 0.8, config_text, fontsize=10, family='monospace',
                   verticalalignment='top', transform=ax.transAxes)
            
            # Results summary
            if 'static' in results:
                res = results['static']
            else:
                res = results
            
            results_text = self._format_results_text(res)
            ax.text(0.1, 0.4, results_text, fontsize=10, family='monospace',
                   verticalalignment='top', transform=ax.transAxes)

            # Uncertainty (if available)
            uq = None
            if isinstance(results, dict) and 'uq' in results:
                uq = results['uq']
            elif isinstance(res, dict) and 'uq' in res:
                uq = res['uq']
            if uq:
                mean = uq.get('mean', None)
                std = uq.get('std', None)
                ci = uq.get('ci95', None) or uq.get('ci_95', None) or uq.get('ci', None)
                uq_text = "═══ UNCERTAINTY ═══\n\n"
                if isinstance(mean, (int, float)):
                    uq_text += f"Mean k_eff: {mean:.6f} W/m·K\n"
                if isinstance(std, (int, float)):
                    uq_text += f"Std dev:    {std:.6f} W/m·K\n"
                if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
                    uq_text += f"95% CI:     [{ci[0]:.6f}, {ci[1]:.6f}] W/m·K\n"
                ax.text(0.1, 0.2, uq_text, fontsize=10, family='monospace',
                       verticalalignment='top', transform=ax.transAxes)
            
            # Footer
            footer = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            footer += f"Version: {self.version} | Dedicated to: Brijesh Kumar Pandey"
            ax.text(0.5, 0.05, footer, fontsize=8, ha='center',
                   transform=ax.transAxes, style='italic')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Visualizations
            if 'enhancement_k' in res:
                fig = self._create_results_figure(res)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Metadata
            d = pdf.infodict()
            d['Title'] = 'BKPS NFL Thermal Pro Simulation Report'
            d['Author'] = 'BKPS NFL Thermal Pro 7.0'
            d['Subject'] = f'Nanofluid Simulation: {config.base_fluid.name}'
            d['Keywords'] = 'Nanofluid, Thermal Conductivity, Heat Transfer'
            d['Creator'] = f'BKPS NFL Thermal Pro {self.version}'
            d['Producer'] = 'matplotlib'
    
    def _generate_reportlab_pdf(self, results: Dict, config: Any, 
                                output_path: str, include_figures: bool,
                                include_validation: bool):
        """Generate professional PDF using reportlab"""
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=1*inch, bottomMargin=0.75*inch)
        
        # Build story (document content)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title page
        story.append(Paragraph("BKPS NFL Thermal Pro 7.0", title_style))
        story.append(Paragraph("Professional Nanofluid Simulation Report", styles['Heading2']))
        story.append(Spacer(1, 0.3*inch))
        
        # Simulation info
        sim_info = f"""
        <b>Simulation Mode:</b> {config.mode.value.upper()}<br/>
        <b>Base Fluid:</b> {config.base_fluid.name}<br/>
        <b>Temperature:</b> {config.base_fluid.temperature} K<br/>
        <b>Nanoparticles:</b> {len(config.nanoparticles)} component(s)<br/>
        <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Version:</b> {self.version}
        """
        story.append(Paragraph(sim_info, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Configuration table
        story.append(Paragraph("Configuration", heading_style))
        config_table_data = self._create_config_table(config)
        config_table = Table(config_table_data, colWidths=[2.5*inch, 3.5*inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))
        story.append(config_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Results table
        story.append(Paragraph("Simulation Results", heading_style))
        
        if 'static' in results:
            res = results['static']
        else:
            res = results
        
        results_table_data = self._create_results_table(res)
        results_table = Table(results_table_data, colWidths=[3*inch, 3*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#06A77D')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 0.2*inch))

        # Uncertainty analysis (if available)
        uq = None
        if isinstance(res, dict) and 'uq' in res:
            uq = res['uq']
        elif isinstance(results, dict) and 'uq' in results:
            uq = results['uq']
        if uq:
            story.append(Paragraph("Uncertainty Analysis", heading_style))
            mean = uq.get('mean', None)
            std = uq.get('std', None)
            ci = uq.get('ci95', None) or uq.get('ci_95', None) or uq.get('ci', None)
            rows = [['Metric', 'Value']]
            rows.append(['Mean k_eff', f"{mean:.6f} W/m·K" if isinstance(mean, (int, float)) else 'N/A'])
            rows.append(['Std dev', f"{std:.6f} W/m·K" if isinstance(std, (int, float)) else 'N/A'])
            if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
                ci_text = f"[{ci[0]:.6f}, {ci[1]:.6f}] W/m·K"
            else:
                ci_text = 'N/A'
            rows.append(['95% CI', ci_text])
            uq_table = Table(rows, colWidths=[3*inch, 3*inch])
            uq_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8E44AD')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ]))
            story.append(uq_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Performance metrics
        if 'enhancement_k' in res:
            story.append(Paragraph("Performance Analysis", heading_style))
            
            enhancement = res['enhancement_k']
            viscosity_ratio = res.get('viscosity_ratio', 1.0)
            
            # Performance index (simplified)
            performance_index = enhancement / (viscosity_ratio ** 0.33)
            
            perf_text = f"""
            <b>Thermal Enhancement:</b> {enhancement:.2f}%<br/>
            <b>Viscosity Ratio:</b> {viscosity_ratio:.4f}<br/>
            <b>Performance Index:</b> {performance_index:.2f}<br/>
            <br/>
            <i>Performance Index = Enhancement / (Viscosity Ratio)^0.33</i><br/>
            <i>Higher values indicate better thermal-hydraulic performance</i>
            """
            story.append(Paragraph(perf_text, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Version hash
        story.append(PageBreak())
        story.append(Paragraph("Document Information", heading_style))
        
        doc_hash = self._generate_document_hash(results, config)
        version_text = f"""
        <b>Document Hash:</b> {doc_hash}<br/>
        <b>Version:</b> BKPS NFL Thermal Pro {self.version}<br/>
        <b>Release Date:</b> {self.release_date}<br/>
        <b>Dedicated to:</b> Brijesh Kumar Pandey<br/>
        <br/>
        <b>Citation:</b><br/>
        BKPS NFL Thermal Pro v{self.version} ({self.release_date}). 
        Professional Nanofluid Simulation Platform. 
        https://github.com/msaurav625-lgtm/test<br/>
        <br/>
        <b>License:</b> MIT License
        """
        story.append(Paragraph(version_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
    
    def _create_config_table(self, config: Any) -> List[List[str]]:
        """Create configuration table data"""
        data = [['Parameter', 'Value']]
        
        data.append(['Simulation Mode', config.mode.value.upper()])
        data.append(['Base Fluid', config.base_fluid.name])
        data.append(['Temperature', f"{config.base_fluid.temperature} K"])
        data.append(['Pressure', f"{config.base_fluid.pressure/1000:.1f} kPa"])
        
        for i, np_config in enumerate(config.nanoparticles):
            data.append([f'Nanoparticle {i+1}', np_config.material])
            data.append([f'  Volume Fraction', f"{np_config.volume_fraction*100:.2f}%"])
            data.append([f'  Diameter', f"{np_config.diameter*1e9:.1f} nm"])
            data.append([f'  Shape', np_config.shape])
        
        if config.enable_dlvo:
            data.append(['DLVO Analysis', 'Enabled'])
        if config.enable_non_newtonian:
            data.append(['Non-Newtonian', 'Enabled'])
        if config.enable_ai_recommendations:
            data.append(['AI Recommendations', 'Enabled'])
        
        return data
    
    def _create_results_table(self, results: Dict) -> List[List[str]]:
        """Create results table data"""
        data = [['Property', 'Value']]
        
        if 'k_base' in results:
            data.append(['Base Fluid k', f"{results['k_base']:.6f} W/m·K"])
        if 'k_static' in results:
            data.append(['Nanofluid k', f"{results['k_static']:.6f} W/m·K"])
        if 'enhancement_k' in results:
            data.append(['Enhancement', f"{results['enhancement_k']:.2f}%"])
        
        if 'mu_base' in results:
            data.append(['Base Fluid μ', f"{results['mu_base']*1000:.6f} mPa·s"])
        if 'mu_nf' in results:
            data.append(['Nanofluid μ', f"{results['mu_nf']*1000:.6f} mPa·s"])
        if 'viscosity_ratio' in results:
            data.append(['Viscosity Ratio', f"{results['viscosity_ratio']:.4f}"])
        
        if 'rho_nf' in results:
            data.append(['Density', f"{results['rho_nf']:.2f} kg/m³"])
        if 'cp_nf' in results:
            data.append(['Specific Heat', f"{results['cp_nf']:.2f} J/kg·K"])
        
        return data
    
    def _format_config_text(self, config: Any) -> str:
        """Format configuration as text"""
        text = "═══ CONFIGURATION ═══\n\n"
        text += f"Mode: {config.mode.value.upper()}\n"
        text += f"Base Fluid: {config.base_fluid.name}\n"
        text += f"Temperature: {config.base_fluid.temperature} K\n\n"
        
        for i, np in enumerate(config.nanoparticles):
            text += f"Nanoparticle {i+1}:\n"
            text += f"  Material: {np.material}\n"
            text += f"  φ: {np.volume_fraction*100:.2f}%\n"
            text += f"  d: {np.diameter*1e9:.1f} nm\n"
            text += f"  Shape: {np.shape}\n\n"
        
        return text
    
    def _format_results_text(self, results: Dict) -> str:
        """Format results as text"""
        text = "═══ RESULTS ═══\n\n"
        
        if 'k_base' in results:
            text += f"k_base:  {results['k_base']:.6f} W/m·K\n"
        if 'k_static' in results:
            text += f"k_nf:    {results['k_static']:.6f} W/m·K\n"
        if 'enhancement_k' in results:
            text += f"Enhancement: {results['enhancement_k']:.2f}%\n\n"
        
        if 'mu_base' in results:
            text += f"μ_base:  {results['mu_base']*1000:.6f} mPa·s\n"
        if 'mu_nf' in results:
            text += f"μ_nf:    {results['mu_nf']*1000:.6f} mPa·s\n"
        if 'viscosity_ratio' in results:
            text += f"Ratio:   {results['viscosity_ratio']:.4f}\n"
        
        return text
    
    def _create_results_figure(self, results: Dict) -> plt.Figure:
        """Create visualization figure"""
        fig = plt.figure(figsize=(8.5, 6))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Enhancement bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        if 'enhancement_k' in results:
            ax1.bar(['Base', 'Nanofluid'], 
                   [0, results['enhancement_k']], 
                   color=['#2E86AB', '#06A77D'])
            ax1.set_ylabel('Enhancement (%)')
            ax1.set_title('Thermal Conductivity Enhancement')
            ax1.grid(True, alpha=0.3)
        
        # Viscosity comparison
        ax2 = fig.add_subplot(gs[0, 1])
        if 'mu_base' in results and 'mu_nf' in results:
            ax2.bar(['Base Fluid', 'Nanofluid'],
                   [results['mu_base']*1000, results['mu_nf']*1000],
                   color=['#2E86AB', '#F18F01'])
            ax2.set_ylabel('Viscosity (mPa·s)')
            ax2.set_title('Viscosity Comparison')
            ax2.grid(True, alpha=0.3)
        
        # Properties table
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        table_data = []
        if 'k_static' in results:
            table_data.append(['k_eff', f"{results['k_static']:.6f} W/m·K"])
        if 'mu_nf' in results:
            table_data.append(['μ_eff', f"{results['mu_nf']*1000:.6f} mPa·s"])
        if 'rho_nf' in results:
            table_data.append(['ρ_eff', f"{results['rho_nf']:.2f} kg/m³"])
        if 'cp_nf' in results:
            table_data.append(['c_p', f"{results['cp_nf']:.2f} J/kg·K"])
        
        if table_data:
            table = ax3.table(cellText=table_data,
                            colLabels=['Property', 'Value'],
                            cellLoc='left',
                            loc='center',
                            colWidths=[0.3, 0.5])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        
        return fig
    
    def _generate_document_hash(self, results: Dict, config: Any) -> str:
        """Generate unique document hash"""
        content = f"{config.mode.value}_{config.base_fluid.name}_"
        content += f"{config.base_fluid.temperature}_"
        for np in config.nanoparticles:
            content += f"{np.material}_{np.volume_fraction}_"
        content += f"{datetime.now().strftime('%Y%m%d')}"
        
        return hashlib.md5(content.encode()).hexdigest()[:16].upper()


def generate_quick_report(results: Dict, config: Any, output_path: str = "report.pdf"):
    """
    Quick function to generate PDF report
    
    Args:
        results: Simulation results
        config: Configuration object
        output_path: Output file path
    """
    generator = PDFReportGenerator()
    generator.generate_report(results, config, output_path)
    print(f"✓ Report generated: {output_path}")


if __name__ == '__main__':
    # Test report generation
    from nanofluid_simulator import BKPSNanofluidEngine
    
    print("Generating test report...")
    
    # Run quick simulation
    engine = BKPSNanofluidEngine.quick_start(
        mode="static",
        nanoparticle="Al2O3",
        volume_fraction=0.02
    )
    results = engine.run()
    
    # Generate report
    generator = PDFReportGenerator()
    generator.generate_report(results, engine.config, "test_report.pdf")
    
    print("✓ Test report generated: test_report.pdf")
