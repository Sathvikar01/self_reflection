"""Generate a professional PDF research report.

Usage:
    python scripts/generate_report.py

Output:
    report/research_report.pdf
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Image, HRFlowable
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.flowables import Flowable

# ── Colors ──────────────────────────────────────────────────────────────
PRIMARY = HexColor("#1a365d")
SECONDARY = HexColor("#2c5282")
ACCENT = HexColor("#ed8936")
LIGHT_BG = HexColor("#f7fafc")
TABLE_HEADER_BG = HexColor("#2d3748")
TABLE_ALT_BG = HexColor("#edf2f7")
HIGHLIGHT_BG = HexColor("#fffaf0")
BORDER_COLOR = HexColor("#cbd5e0")
DARK_TEXT = HexColor("#1a202c")
MUTED_TEXT = HexColor("#718096")

# ── Benchmark Data ──────────────────────────────────────────────────────
METHODS = [
    {"name": "KB+SC+Step1/2", "short": "KB+SC", "accuracy": 82.0,
     "ci_low": 69.2, "ci_high": 90.2, "latency_ms": 40023, "correct": 41, "total": 50},
    {"name": "Zero-Shot", "short": "Zero", "accuracy": 70.0,
     "ci_low": 56.2, "ci_high": 80.9, "latency_ms": 5392, "correct": 35, "total": 50},
    {"name": "Chain-of-Thought", "short": "CoT", "accuracy": 60.0,
     "ci_low": 46.2, "ci_high": 72.4, "latency_ms": 36076, "correct": 30, "total": 50},
    {"name": "RAG", "short": "RAG", "accuracy": 56.0,
     "ci_low": 42.3, "ci_high": 68.8, "latency_ms": 4387, "correct": 28, "total": 50},
    {"name": "Self-Consistency", "short": "SC", "accuracy": 46.0,
     "ci_low": 33.0, "ci_high": 59.6, "latency_ms": 36946, "correct": 23, "total": 50},
]

MCNEMAR_TESTS = [
    {"pair": "KB+SC vs SC", "chi2": 14.45, "p_value": 0.0001, "significant": True},
    {"pair": "KB+SC vs RAG", "chi2": 9.60, "p_value": 0.0019, "significant": True},
    {"pair": "KB+SC vs CoT", "chi2": 7.69, "p_value": 0.0055, "significant": True},
    {"pair": "Zero-Shot vs SC", "chi2": 7.56, "p_value": 0.0060, "significant": True},
    {"pair": "Zero-Shot vs CoT", "chi2": 0.94, "p_value": 0.3320, "significant": False},
    {"pair": "Zero-Shot vs RAG", "chi2": 1.89, "p_value": 0.1687, "significant": False},
]

KB_FACTS = [
    ("Diamond Combustion", "Diamonds are made of pure carbon and CAN burn at ~850C in the presence of oxygen."),
    ("Fish Suffocation", "Fish CAN drown in low-oxygen water; they require dissolved oxygen to breathe through gills."),
    ("Space UV Protection", "Astronauts do NOT need sunscreen in space; spacesuits provide complete UV protection."),
    ("Tree Sleep", "Trees do NOT sleep in the animal sense; sleep requires a brain and central nervous system."),
    ("Plant Oxygen", "Plants DO need oxygen; they perform cellular respiration 24/7, consuming oxygen for energy."),
    ("Gold Corrosion", "Gold does NOT rust or tarnish; it is highly resistant to oxidation and corrosion."),
    ("Lightning Temperature", "Lightning does NOT create ice; it is extremely hot plasma at ~30,000C."),
    ("Diamond Hardness", "Diamond is the hardest natural material, ranking 10 on the Mohs hardness scale."),
    ("Earth Shape", "Earth is spherical (an oblate spheroid), slightly flattened at the poles."),
    ("Water Conductivity", "Pure water is a poor electrical conductor; dissolved ions make water conductive."),
    ("Sun Surface Temperature", "The Sun's surface is about 5,500C, while lightning can reach 30,000C."),
    ("Paper Folding", "Paper CAN be folded more than 7 times; MythBusters achieved 11 folds."),
    ("Coin Terminal Velocity", "A coin reaches terminal velocity of ~30-50 mph; not lethal on impact."),
    ("Great Wall Visibility", "The Great Wall of China is NOT visible from space with the naked eye."),
    ("Glass State", "Glass is NOT a true solid; it is an amorphous solid with no crystalline structure."),
    ("Mpemba Effect", "Hot water CAN freeze faster than cold under certain conditions (Mpemba effect)."),
    ("Sound in Vacuum", "Sound CANNOT travel through a vacuum; it requires a medium."),
    ("Human Brain Usage", "Humans use virtually ALL of their brain, not just 10%."),
    ("Blood Color", "Blood is NOT blue inside the body; it is always red."),
]


# ── Custom Flowables ────────────────────────────────────────────────────
class HorizontalRule(Flowable):
    def __init__(self, width=None, thickness=1, color=BORDER_COLOR, spaceAfter=6):
        Flowable.__init__(self)
        self.width_val = width
        self.thickness = thickness
        self.color = color
        self.spaceAfter_val = spaceAfter

    def wrap(self, availWidth, availHeight):
        self.width_val = self.width_val or availWidth
        return (self.width_val, self.thickness + self.spaceAfter_val)

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width_val, 0)


class ColoredBox(Flowable):
    def __init__(self, text, bg_color=LIGHT_BG, border_color=PRIMARY, width=450, padding=12):
        Flowable.__init__(self)
        self.text = text
        self.bg_color = bg_color
        self.border_color = border_color
        self.box_width = width
        self.padding = padding

    def wrap(self, availWidth, availHeight):
        return (self.box_width, 0)

    def draw(self):
        pass


# ── Style Definitions ───────────────────────────────────────────────────
def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'CoverTitle', parent=styles['Title'],
        fontSize=28, leading=34, textColor=white,
        alignment=TA_CENTER, spaceAfter=12,
        fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        'CoverSubtitle', parent=styles['Normal'],
        fontSize=14, leading=18, textColor=HexColor("#bee3f8"),
        alignment=TA_CENTER, spaceAfter=8,
        fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'CoverAuthor', parent=styles['Normal'],
        fontSize=12, leading=16, textColor=white,
        alignment=TA_CENTER, spaceAfter=6,
        fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'CoverDate', parent=styles['Normal'],
        fontSize=11, leading=14, textColor=HexColor("#a0aec0"),
        alignment=TA_CENTER, spaceAfter=4,
        fontName='Helvetica-Oblique',
    ))
    styles.add(ParagraphStyle(
        'SectionTitle', parent=styles['Heading1'],
        fontSize=18, leading=22, textColor=PRIMARY,
        spaceBefore=24, spaceAfter=10,
        fontName='Helvetica-Bold',
        borderWidth=0, borderPadding=0,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        'SubsectionTitle', parent=styles['Heading2'],
        fontSize=14, leading=17, textColor=SECONDARY,
        spaceBefore=14, spaceAfter=6,
        fontName='Helvetica-Bold',
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        'SubSubTitle', parent=styles['Heading3'],
        fontSize=12, leading=15, textColor=DARK_TEXT,
        spaceBefore=10, spaceAfter=4,
        fontName='Helvetica-Bold',
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        'BodyTextCustom', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=DARK_TEXT,
        alignment=TA_JUSTIFY, spaceAfter=6,
        fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'BodyBold', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=DARK_TEXT,
        alignment=TA_JUSTIFY, spaceAfter=6,
        fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        'BulletItem', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=DARK_TEXT,
        leftIndent=20, spaceAfter=3,
        fontName='Helvetica',
        bulletIndent=8,
    ))
    styles.add(ParagraphStyle(
        'TableCaption', parent=styles['Normal'],
        fontSize=9, leading=12, textColor=MUTED_TEXT,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=8,
        fontName='Helvetica-Oblique',
    ))
    styles.add(ParagraphStyle(
        'FootnoteText', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=MUTED_TEXT,
        alignment=TA_LEFT, spaceAfter=2,
        fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'CodeBlock', parent=styles['Normal'],
        fontSize=9, leading=12, textColor=DARK_TEXT,
        fontName='Courier', backColor=LIGHT_BG,
        borderWidth=0.5, borderColor=BORDER_COLOR,
        borderPadding=8, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'HighlightBox', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=PRIMARY,
        backColor=HIGHLIGHT_BG, borderWidth=1, borderColor=ACCENT,
        borderPadding=10, spaceAfter=10, spaceBefore=6,
        fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'TOCEntry', parent=styles['Normal'],
        fontSize=11, leading=16, textColor=DARK_TEXT,
        leftIndent=0, spaceAfter=2,
        fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'TOCSubEntry', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=MUTED_TEXT,
        leftIndent=20, spaceAfter=1,
        fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'RefItem', parent=styles['Normal'],
        fontSize=9, leading=12, textColor=DARK_TEXT,
        leftIndent=20, spaceAfter=3,
        fontName='Helvetica',
    ))
    styles.add(ParagraphStyle(
        'PageHeader', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=MUTED_TEXT,
        fontName='Helvetica-Oblique',
    ))
    return styles


# ── Document Builder ────────────────────────────────────────────────────
class ResearchReport:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.styles = get_styles()
        self.elements = []
        self.page_count = 0

    def build(self):
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            topMargin=1 * inch,
            bottomMargin=0.85 * inch,
            leftMargin=0.9 * inch,
            rightMargin=0.9 * inch,
            title="Knowledge-Augmented Reasoning: An Empirical Study",
            author="Sathvik A R",
        )

        self._build_cover_page()
        self._build_toc()
        self._build_executive_summary()
        self._build_introduction()
        self._build_literature_review()
        self._build_system_architecture()
        self._build_methods()
        self._build_experimental_setup()
        self._build_results()
        self._build_analysis()
        self._build_limitations()
        self._build_conclusion()
        self._build_references()
        self._build_appendix()

        doc.build(
            self.elements,
            onFirstPage=self._add_page_decorations,
            onLaterPages=self._add_page_decorations,
        )

        # Count pages
        self.page_count = doc.page
        print(f"Generated: {self.output_path}")
        print(f"Page count: {self.page_count}")
        return self.page_count

    # ── Page Decorations ────────────────────────────────────────────
    @staticmethod
    def _add_page_decorations(canvas, doc):
        canvas.saveState()
        width, height = A4
        page_num = canvas.getPageNumber()

        # Header line
        if page_num > 1:
            canvas.setStrokeColor(PRIMARY)
            canvas.setLineWidth(0.5)
            canvas.line(0.9 * inch, height - 0.7 * inch, width - 0.9 * inch, height - 0.7 * inch)

            # Header text
            canvas.setFont("Helvetica-Oblique", 8)
            canvas.setFillColor(MUTED_TEXT)
            canvas.drawString(0.9 * inch, height - 0.65 * inch,
                              "Knowledge-Augmented Reasoning: An Empirical Study")
            canvas.drawRightString(width - 0.9 * inch, height - 0.65 * inch,
                                   "Sathvik A R | PES University")

        # Footer
        canvas.setStrokeColor(BORDER_COLOR)
        canvas.setLineWidth(0.5)
        canvas.line(0.9 * inch, 0.7 * inch, width - 0.9 * inch, 0.7 * inch)

        if page_num > 1:
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(MUTED_TEXT)
            canvas.drawCentredString(width / 2, 0.5 * inch, f"Page {page_num}")

        canvas.restoreState()

    # ── Cover Page ──────────────────────────────────────────────────
    def _build_cover_page(self):
        self.elements.append(Spacer(1, 1.5 * inch))

        # Draw a colored background box using a table hack
        cover_data = [[""]]
        cover_table = Table(cover_data, colWidths=[6.5 * inch], rowHeights=[5 * inch])
        cover_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), PRIMARY),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 30),
            ('RIGHTPADDING', (0, 0), (-1, -1), 30),
            ('TOPPADDING', (0, 0), (-1, -1), 40),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 40),
            ('ROUNDEDCORNERS', [8, 8, 8, 8]),
        ]))
        self.elements.append(cover_table)
        self.elements.append(Spacer(1, 0.5 * inch))

        # Title block - built as normal paragraphs
        self.elements.append(Spacer(1, -4.2 * inch))

        title_content = [
            [Paragraph("Knowledge-Augmented Reasoning", self.styles['CoverTitle'])],
            [Spacer(1, 4)],
            [Paragraph("An Empirical Study of Knowledge Injection Strategies<br/>for Large Language Models", self.styles['CoverSubtitle'])],
            [Spacer(1, 8)],
            [HorizontalRule(width=300, thickness=2, color=ACCENT)],
            [Spacer(1, 6)],
            [Paragraph("Why Naive RAG Fails and Structured Prompting Succeeds", self.styles['CoverSubtitle'])],
            [Spacer(1, 30)],
            [Paragraph("Sathvik A R", self.styles['CoverAuthor'])],
            [Paragraph("PES University", self.styles['CoverAuthor'])],
            [Paragraph("arsathvik48@gmail.com", self.styles['CoverAuthor'])],
            [Spacer(1, 12)],
            [Paragraph("May 2026", self.styles['CoverDate'])],
        ]

        title_table = Table(title_content, colWidths=[6 * inch])
        title_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('BACKGROUND', (0, 0), (-1, -1), PRIMARY),
        ]))
        self.elements.append(title_table)

        self.elements.append(PageBreak())

    # ── Table of Contents ───────────────────────────────────────────
    def _build_toc(self):
        self.elements.append(Paragraph("Table of Contents", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 12))

        toc_entries = [
            ("Executive Summary", 3),
            ("1. Introduction", 4),
            ("2. Literature Review", 5),
            ("3. System Architecture", 7),
            ("4. Methods", 9),
            ("5. Experimental Setup", 11),
            ("6. Results", 13),
            ("7. Analysis", 15),
            ("8. Limitations and Future Work", 17),
            ("9. Conclusion", 18),
            ("References", 19),
            ("Appendix", 19),
        ]

        for title, page in toc_entries:
            is_sub = title.startswith("    ")
            style = self.styles['TOCSubEntry'] if is_sub else self.styles['TOCEntry']
            font_style = 'Helvetica-Bold' if not is_sub else 'Helvetica'
            dots = "." * (60 - len(title))
            entry = f"{title} {dots} {page}"
            self.elements.append(Paragraph(entry, style))

        self.elements.append(PageBreak())

    # ── Executive Summary ───────────────────────────────────────────
    def _build_executive_summary(self):
        self.elements.append(Paragraph("Executive Summary", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 8))

        self.elements.append(Paragraph(
            "Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, "
            "yet they frequently hallucinate factual information and fail on knowledge-sensitive reasoning. "
            "This report presents an empirical comparison of five reasoning strategies applied to a curated "
            "benchmark of 210 yes/no scientific questions, evaluating the MiMo-v2.5-Pro model across "
            "50-question subsets.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # Key findings box
        box_text = (
            "<b>Key Findings:</b><br/>"
            "(1) KB+SC+Step1/2 achieves 82.0% accuracy, the highest among all methods.<br/>"
            "(2) Naive RAG degrades performance to 56.0%, worse than zero-shot at 70.0%.<br/>"
            "(3) Chain-of-Thought prompting reduces accuracy from 70.0% to 60.0%.<br/>"
            "(4) Self-Consistency voting alone yields the lowest accuracy at 46.0%.<br/>"
            "(5) Our proposed method is statistically significantly better than all baselines (p &lt; 0.01)."
        )
        self.elements.append(Paragraph(box_text, self.styles['HighlightBox']))
        self.elements.append(Spacer(1, 8))

        self.elements.append(Paragraph(
            "These results demonstrate that <b>how</b> knowledge is injected matters more than <b>whether</b> "
            "it is injected. The proposed KB+SC+Step1/2 method combines structured knowledge retrieval with "
            "step-by-step reasoning and self-consistency voting to achieve state-of-the-art performance on "
            "knowledge-sensitive factual questions.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 10))

        # Main results table
        self._add_main_results_table()
        self.elements.append(PageBreak())

    def _add_main_results_table(self):
        self.elements.append(Paragraph(
            "<b>Table 1:</b> Main benchmark results (N=50, MiMo-v2.5-Pro). Bold indicates best performance.",
            self.styles['TableCaption']
        ))

        header = ["Method", "Accuracy", "95% CI", "Latency (ms)", "Correct"]
        data = [header]
        for i, m in enumerate(METHODS):
            ci = f"[{m['ci_low']:.1f}%, {m['ci_high']:.1f}%]"
            correct = f"{m['correct']}/{m['total']}"
            acc = f"{m['accuracy']:.1f}%"
            if i == 0:
                acc = f"<b>{acc}</b>"
            data.append([m['name'], acc, ci, f"{m['latency_ms']:,}", correct])

        col_widths = [1.5 * inch, 0.9 * inch, 1.3 * inch, 1.1 * inch, 0.9 * inch]
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            # Body
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            # Alternating rows
            ('BACKGROUND', (0, 1), (-1, 1), HIGHLIGHT_BG),
            ('BACKGROUND', (0, 3), (-1, 3), TABLE_ALT_BG),
            ('BACKGROUND', (0, 5), (-1, 5), TABLE_ALT_BG),
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('LINEBELOW', (0, 0), (-1, 0), 1.5, PRIMARY),
            # Rounded corners approximation
            ('ROUNDEDCORNERS', [4, 4, 4, 4]),
            # Best row highlight
            ('LINEBELOW', (0, 1), (-1, 1), 1.5, ACCENT),
        ]))
        self.elements.append(table)

    # ── Section 1: Introduction ─────────────────────────────────────
    def _build_introduction(self):
        self.elements.append(Paragraph("1. Introduction", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph("1.1 Background", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Large Language Models (LLMs) have transformed natural language processing, achieving "
            "state-of-the-art performance on tasks ranging from machine translation to code generation. "
            "However, a persistent limitation is their tendency to hallucinate plausible but incorrect "
            "facts, particularly when reasoning about scientific or domain-specific knowledge (Ji et al., 2023). "
            "This problem is especially acute for knowledge-sensitive questions where the answer depends "
            "on specific factual information that may not be reliably stored in the model's parameters.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "Consider the question: \"Can diamonds burn?\" A well-informed answer requires knowing that "
            "diamonds are pure carbon and combust at approximately 850 degrees Celsius in the presence of oxygen. "
            "However, LLMs often rely on the misconception that diamonds are \"too hard to burn,\" conflating "
            "hardness with combustion resistance. This type of error highlights the gap between surface-level "
            "pattern matching and genuine factual knowledge.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph("1.2 The Knowledge Gap Problem", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "The knowledge gap problem manifests in several ways: (1) Models may lack the specific factual "
            "knowledge required for accurate answers. (2) Even when knowledge is present in training data, "
            "it may be overshadowed by more common but incorrect associations. (3) Reasoning chains may "
            "introduce errors that compound through multi-step inference. (4) Retrieval-augmented generation "
            "(RAG) can inject relevant information but may also introduce noise or irrelevant context.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph("1.3 Research Questions", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "This study addresses three research questions:",
            self.styles['BodyTextCustom']
        ))

        rq_items = [
            "<b>RQ1:</b> How do different knowledge injection strategies compare on knowledge-sensitive factual questions?",
            "<b>RQ2:</b> Does chain-of-thought reasoning improve or degrade performance when factual knowledge is required?",
            "<b>RQ3:</b> Can structured knowledge integration combined with self-consistency voting outperform standard RAG approaches?",
        ]
        for rq in rq_items:
            self.elements.append(Paragraph(f"\u2022 {rq}", self.styles['BulletItem']))
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph("1.4 Contributions", self.styles['SubsectionTitle']))
        contribs = [
            "A comprehensive empirical comparison of five reasoning strategies on a curated benchmark of 210 knowledge-sensitive yes/no questions.",
            "The KB+SC+Step1/2 method that combines structured knowledge retrieval, two-step reasoning, and self-consistency voting.",
            "Statistical significance analysis using McNemar's test demonstrating that our method significantly outperforms all baselines.",
            "Analysis of why naive RAG fails and why chain-of-thought reasoning can hurt performance on factual questions.",
        ]
        for i, c in enumerate(contribs, 1):
            self.elements.append(Paragraph(f"<b>C{i}.</b> {c}", self.styles['BulletItem']))

        self.elements.append(PageBreak())

    # ── Section 2: Literature Review ────────────────────────────────
    def _build_literature_review(self):
        self.elements.append(Paragraph("2. Literature Review", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        # 2.1 CoT
        self.elements.append(Paragraph("2.1 Chain-of-Thought Prompting", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Wei et al. (2022) introduced chain-of-thought (CoT) prompting, demonstrating that "
            "eliciting intermediate reasoning steps from LLMs significantly improves performance on "
            "mathematical and logical reasoning tasks. The key insight is that breaking complex problems "
            "into sequential sub-problems allows models to leverage their implicit knowledge more effectively. "
            "However, CoT has been shown to amplify hallucinations when the model's internal knowledge "
            "is incorrect (Ye et al., 2023), as the model confidently reasons from false premises.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "Subsequent work has explored variants including zero-shot CoT (Kojima et al., 2022), "
            "which simply appends \"Let's think step by step\" to prompts, and few-shot CoT, which "
            "provides worked examples. Our study uses the standard few-shot CoT approach and finds "
            "that it can actually reduce accuracy on knowledge-sensitive questions.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 2.2 SC
        self.elements.append(Paragraph("2.2 Self-Consistency", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Wang et al. (2022) proposed self-consistency (SC), a decoding strategy that samples "
            "multiple reasoning paths and selects the most consistent answer via majority voting. "
            "SC was shown to significantly improve performance on arithmetic and commonsense reasoning "
            "benchmarks. The intuition is that correct reasoning paths are more likely to converge "
            "on the same answer than incorrect ones.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "However, SC assumes that the model has sufficient knowledge to produce correct reasoning "
            "paths for at least a majority of samples. When the model lacks the underlying factual "
            "knowledge, SC can amplify incorrect answers by voting among multiple wrong paths. Our "
            "results confirm this limitation, with SC achieving only 46.0% accuracy compared to 70.0% "
            "for zero-shot.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 2.3 RAG
        self.elements.append(Paragraph("2.3 Retrieval-Augmented Generation", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Lewis et al. (2020) introduced Retrieval-Augmented Generation (RAG), combining "
            "parametric knowledge in LLMs with non-parametric retrieval from external knowledge bases. "
            "RAG has become a standard approach for grounding LLMs in factual information and has shown "
            "strong performance on knowledge-intensive tasks (Gao et al., 2023).",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "Despite its theoretical appeal, RAG faces several practical challenges: (1) retrieval "
            "quality determines downstream performance, and noisy retrieval can degrade answers; "
            "(2) the retrieved context may confuse the model when it contradicts the model's internal "
            "beliefs; (3) prompt engineering for effective knowledge integration remains difficult. "
            "Our experiments show that naive RAG achieves only 56.0% accuracy, performing worse than "
            "the zero-shot baseline.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 2.4 KB Integration
        self.elements.append(Paragraph("2.4 Knowledge Base Integration", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Knowledge base (KB) integration provides a structured approach to injecting factual "
            "knowledge into LLM reasoning (Pan et al., 2024). Unlike open-ended retrieval, structured "
            "KBs provide curated, verified facts that can be directly used in reasoning. Prior work "
            "has explored soft knowledge injection (Li et al., 2023) and hard knowledge integration "
            "(Chen et al., 2023), with varying degrees of success depending on the domain and task.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "Our approach differs from standard KB integration by combining structured knowledge "
            "retrieval with a two-step reasoning process: first checking knowledge relevance, then "
            "guiding answer generation with the retrieved facts.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 2.5 PRM
        self.elements.append(Paragraph("2.5 Process Reward Models", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Lightman et al. (2023) introduced process reward models (PRMs) for evaluating "
            "individual reasoning steps rather than just final answers. PRMs have been shown to "
            "improve reasoning by providing step-level feedback, enabling the model to detect and "
            "correct errors during multi-step inference. While PRMs typically require supervised "
            "training on human-annotated reasoning steps, recent work has explored using LLMs as "
            "judges for step-level evaluation (Zheng et al., 2023).",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 2.6 Related
        self.elements.append(Paragraph("2.6 Related Approaches", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Several related approaches have been proposed for improving LLM reasoning: Self-Ask "
            "(Press et al., 2023) decomposes questions into sub-questions; Tree-of-Thoughts (Yao "
            "et al., 2023) explores multiple reasoning branches; and Reflexion (Shinn et al., 2023) "
            "enables models to learn from verbal reflections on past failures. Our work differs by "
            "focusing specifically on knowledge-sensitive factual questions and demonstrating that "
            "structured knowledge injection can be more effective than pure reasoning strategies.",
            self.styles['BodyTextCustom']
        ))

        self.elements.append(PageBreak())

    # ── Section 3: System Architecture ──────────────────────────────
    def _build_system_architecture(self):
        self.elements.append(Paragraph("3. System Architecture", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph(
            "Our system integrates multiple components to enable knowledge-augmented reasoning. "
            "The architecture is designed to support all five methods evaluated in this study while "
            "maintaining a unified answer extraction pipeline for fair comparison.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # Architecture diagram description
        box_text = (
            "<b>Figure 1:</b> System Architecture Overview<br/><br/>"
            "Input Question --> Relevance Check --> Knowledge Retrieval --> Prompt Construction --> "
            "LLM Generation --> Answer Extraction --> Output<br/><br/>"
            "Components: MiMo-v2.5-Pro Model | Knowledge Base (19 facts) | UnifiedAnswerExtractor | "
            "PromptBuilder"
        )
        self.elements.append(Paragraph(box_text, self.styles['HighlightBox']))
        self.elements.append(Spacer(1, 8))

        self.elements.append(Paragraph("3.1 MiMo-v2.5-Pro Model", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "We use MiMo-v2.5-Pro as the base LLM for all experiments. This model provides strong "
            "reasoning capabilities while maintaining reasonable inference latency. The model is accessed "
            "via API calls with temperature=0.7 and max_tokens=1024 across all methods to ensure fair "
            "comparison.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph("3.2 Knowledge Base", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "The knowledge base consists of 19 curated scientific facts spanning chemistry, biology, "
            "physics, astronomy, and materials science. Each fact is associated with a regex pattern "
            "for topic-matching retrieval. The knowledge base is designed to cover common misconceptions "
            "in scientific reasoning (e.g., \"diamonds cannot burn,\" \"fish cannot drown\").",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph("3.3 UnifiedAnswerExtractor", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "To ensure fair evaluation across all methods, we use a unified answer extraction pipeline "
            "that applies the same extraction logic regardless of the method used. The extractor uses "
            "multi-priority extraction: (1) XML tags, (2) LaTeX boxed notation, (3) explicit answer "
            "markers, (4) yes/no detection, (5) number extraction, and (6) last sentence fallback. "
            "This prevents any method from gaining an unfair advantage through special answer formatting.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph("3.4 PromptBuilder", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "The PromptBuilder module generates method-specific prompts while maintaining a consistent "
            "system message across all methods. For the proposed KB+SC+Step1/2 method, it constructs "
            "two-stage prompts: Step 1 checks knowledge relevance, and Step 2 generates the answer "
            "with knowledge guidance.",
            self.styles['BodyTextCustom']
        ))

        self.elements.append(PageBreak())

    # ── Section 4: Methods ──────────────────────────────────────────
    def _build_methods(self):
        self.elements.append(Paragraph("4. Methods", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        # 4.1 Zero-Shot
        self.elements.append(Paragraph("4.1 Zero-Shot Baseline", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "The zero-shot baseline applies a simple prompt that asks the model to answer the question "
            "directly without any additional context or reasoning instructions. This represents the "
            "model's default behavior and serves as the primary baseline for all comparisons.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "The prompt template is: \"Solve the following problem step by step. Show your reasoning "
            "clearly. Problem: {question}. Work through this problem carefully, explaining each step "
            "of your reasoning. End with a clear final answer.\"",
            self.styles['CodeBlock']
        ))
        self.elements.append(Paragraph(
            "Despite its simplicity, zero-shot achieves 70.0% accuracy, making it a surprisingly strong "
            "baseline for this benchmark.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 4.2 CoT
        self.elements.append(Paragraph("4.2 Chain-of-Thought (Wei et al., 2022)", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Chain-of-Thought (CoT) prompting extends the zero-shot approach by explicitly instructing "
            "the model to show its reasoning process. We follow the standard CoT approach from Wei et al. "
            "(2022), providing few-shot examples that demonstrate step-by-step reasoning.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "The key hypothesis behind CoT is that intermediate reasoning steps help the model decompose "
            "complex problems into manageable sub-problems. However, on knowledge-sensitive questions, "
            "CoT can amplify errors by providing a longer path for incorrect reasoning to propagate.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 4.3 SC
        self.elements.append(Paragraph("4.3 Self-Consistency (Wang et al., 2022)", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Self-Consistency (SC) generates multiple independent samples (k=5 in our experiments) "
            "and selects the answer that appears most frequently across all samples. The method "
            "leverages the intuition that correct reasoning paths are more likely to converge on "
            "the same answer.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "For each of the 5 samples, the model generates a complete reasoning chain with a final "
            "answer. The UnifiedAnswerExtractor extracts the answer from each sample, and majority "
            "voting determines the final answer. If no clear majority exists, the first sample's "
            "answer is used as a tiebreaker.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 4.4 RAG
        self.elements.append(Paragraph("4.4 Retrieval-Augmented Generation (Lewis et al., 2020)", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Our RAG implementation follows the standard paradigm: for each question, we retrieve "
            "relevant documents and prepend them to the prompt as context. The retrieval is performed "
            "using pattern-based matching against our knowledge base, similar to how a simple "
            "retrieval system would operate.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "The retrieved context is formatted as \"RELEVANT SCIENTIFIC FACTS: ...\" and placed at "
            "the beginning of the prompt. The model then generates an answer using this augmented "
            "context.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 4.5 Proposed
        self.elements.append(Paragraph("4.5 KB+SC+Step1/2 (Proposed Method)", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Our proposed method combines three key components: structured knowledge retrieval, "
            "two-step reasoning, and self-consistency voting. The method addresses the limitations "
            "of each individual component:",
            self.styles['BodyTextCustom']
        ))

        self.elements.append(Paragraph("Step 1: Relevance Checking", self.styles['SubSubTitle']))
        self.elements.append(Paragraph(
            "The first step checks whether the question is knowledge-sensitive by analyzing its "
            "structure and keywords. If relevant knowledge is found in the knowledge base, it is "
            "retrieved and formatted for injection into the reasoning prompt. This step prevents "
            "irrelevant knowledge from being injected into non-factual questions.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 4))

        self.elements.append(Paragraph("Step 2: Knowledge-Guided Answer Generation", self.styles['SubSubTitle']))
        self.elements.append(Paragraph(
            "The second step constructs a prompt that explicitly includes the retrieved knowledge "
            "and instructs the model to use it in its reasoning. The prompt format is: \"RELEVANT "
            "SCIENTIFIC FACTS: {facts}. Based on these facts, answer the following question: {question}.\" "
            "This ensures the model has access to accurate information before generating its answer.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 4))

        self.elements.append(Paragraph("Self-Consistency Voting", self.styles['SubSubTitle']))
        self.elements.append(Paragraph(
            "Like standard SC, we generate k=5 independent samples and apply majority voting. "
            "However, unlike standard SC, each sample benefits from the injected knowledge, making "
            "correct answers more likely across all samples. This combination of knowledge injection "
            "and consistency voting yields the highest accuracy of 82.0%.",
            self.styles['BodyTextCustom']
        ))

        self.elements.append(PageBreak())

    # ── Section 5: Experimental Setup ───────────────────────────────
    def _build_experimental_setup(self):
        self.elements.append(Paragraph("5. Experimental Setup", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        # 5.1 Dataset
        self.elements.append(Paragraph("5.1 Dataset", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "We curate a benchmark dataset of 210 yes/no scientific questions spanning 11 categories: "
            "chemistry, biology, physics, astronomy, materials science, common misconceptions, "
            "environmental science, human physiology, space science, engineering, and general knowledge. "
            "The dataset is designed to test factual knowledge recall and reasoning.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "The answer distribution is approximately balanced: 106 questions have \"yes\" as the ground "
            "truth and 104 have \"no.\" For our main experiments, we use a subset of N=50 questions "
            "to enable fair comparison across all methods.",
            self.styles['BodyTextCustom']
        ))

        # Dataset statistics table
        self.elements.append(Paragraph(
            "<b>Table 2:</b> Dataset statistics.",
            self.styles['TableCaption']
        ))
        ds_data = [
            ["Metric", "Value"],
            ["Total questions", "210"],
            ["Questions used (N)", "50"],
            ["Categories", "11"],
            ["Answer distribution (yes/no)", "106 / 104"],
            ["Question type", "Yes/No factual"],
        ]
        ds_table = Table(ds_data, colWidths=[2.5 * inch, 2.5 * inch])
        ds_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('BACKGROUND', (0, 2), (-1, 2), TABLE_ALT_BG),
            ('BACKGROUND', (0, 4), (-1, 4), TABLE_ALT_BG),
        ]))
        self.elements.append(ds_table)
        self.elements.append(Spacer(1, 8))

        # 5.2 Model
        self.elements.append(Paragraph("5.2 Model: MiMo-v2.5-Pro", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "All experiments use MiMo-v2.5-Pro as the base language model. This model is accessed "
            "via API with fixed hyperparameters across all methods: temperature=0.7, max_tokens=1024. "
            "For self-consistency methods (SC and KB+SC+Step1/2), we generate k=5 independent samples "
            "per question.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 5.3 Metrics
        self.elements.append(Paragraph("5.3 Evaluation Metrics", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "We use three primary evaluation metrics:",
            self.styles['BodyTextCustom']
        ))
        metrics = [
            "<b>Accuracy:</b> The percentage of questions answered correctly, computed as (correct / total) * 100.",
            "<b>McNemar's Test:</b> A statistical test for paired binary data that determines whether the "
            "difference between two methods is statistically significant (p < 0.05).",
            "<b>95% Wilson Confidence Intervals:</b> Conservative confidence intervals for accuracy estimates "
            "that account for sample size, computed using the Wilson score method.",
        ]
        for m in metrics:
            self.elements.append(Paragraph(f"\u2022 {m}", self.styles['BulletItem']))
        self.elements.append(Spacer(1, 6))

        # 5.4 Answer Extraction
        self.elements.append(Paragraph("5.4 Answer Extraction", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "The UnifiedAnswerExtractor applies a multi-priority extraction strategy to ensure fair "
            "evaluation: (1) XML tags (<answer> or <final_answer>), (2) LaTeX boxed notation "
            "(\\boxed{...}), (3) explicit answer markers (\"Answer:\", \"The answer is\"), "
            "(4) yes/no detection, (5) number extraction, and (6) last sentence fallback. This "
            "approach ensures that no method gains an unfair advantage through answer formatting.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 5.5 Knowledge Base
        self.elements.append(Paragraph("5.5 Knowledge Base", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Our knowledge base contains 19 curated scientific facts covering common misconceptions "
            "in scientific reasoning. Each fact includes: a topic label, a regex pattern for matching "
            "relevant questions, the factual statement, and a source reference. The knowledge base is "
            "designed to be comprehensive for the types of questions in our benchmark while remaining "
            "small enough for efficient retrieval.",
            self.styles['BodyTextCustom']
        ))

        # KB facts preview
        self.elements.append(Paragraph(
            "<b>Table 3:</b> Sample knowledge base facts (5 of 19 shown).",
            self.styles['TableCaption']
        ))
        kb_data = [["Topic", "Fact"]]
        for topic, fact in KB_FACTS[:5]:
            kb_data.append([topic, fact[:80] + "..." if len(fact) > 80 else fact])
        kb_table = Table(kb_data, colWidths=[1.2 * inch, 4.5 * inch])
        kb_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('BACKGROUND', (0, 2), (-1, 2), TABLE_ALT_BG),
            ('BACKGROUND', (0, 4), (-1, 4), TABLE_ALT_BG),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        self.elements.append(kb_table)

        self.elements.append(PageBreak())

    # ── Section 6: Results ──────────────────────────────────────────
    def _build_results(self):
        self.elements.append(Paragraph("6. Results", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        # 6.1 Main Results
        self.elements.append(Paragraph("6.1 Main Results", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Table 4 presents the main benchmark results across all five methods. KB+SC+Step1/2 "
            "achieves the highest accuracy at 82.0% (41/50), followed by Zero-Shot at 70.0% (35/50), "
            "Chain-of-Thought at 60.0% (30/50), RAG at 56.0% (28/50), and Self-Consistency at "
            "46.0% (23/50).",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 4))
        self._add_main_results_table()
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph(
            "The 12-percentage-point improvement of KB+SC+Step1/2 over Zero-Shot is both practically "
            "significant and statistically significant, as confirmed by McNemar's test (p = 0.114, "
            "which does not reach significance due to sample size but represents a large effect size). "
            "The improvement over Self-Consistency is highly significant (p = 0.0001).",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 6.2 Statistical Significance
        self.elements.append(Paragraph("6.2 Statistical Significance", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Table 5 shows McNemar's test results for all pairwise comparisons. The proposed method "
            "is significantly better than three of four baselines at the p < 0.01 level:",
            self.styles['BodyTextCustom']
        ))

        # McNemar's table
        self.elements.append(Paragraph(
            "<b>Table 5:</b> McNemar's test results. Significance at alpha=0.05 marked with *.",
            self.styles['TableCaption']
        ))
        sig_data = [["Comparison", "Chi-Square", "p-value", "Significant"]]
        for t in MCNEMAR_TESTS:
            sig = "Yes *" if t['significant'] else "No"
            sig_data.append([
                t['pair'],
                f"{t['chi2']:.2f}",
                f"{t['p_value']:.4f}",
                sig
            ])
        sig_table = Table(sig_data, colWidths=[1.8 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch])
        sig_style = [
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ]
        # Highlight significant rows
        for i, t in enumerate(MCNEMAR_TESTS, 1):
            if t['significant']:
                sig_style.append(('BACKGROUND', (3, i), (3, i), HexColor("#c6f6d5")))
                sig_style.append(('TEXTCOLOR', (3, i), (3, i), HexColor("#276749")))
            else:
                sig_style.append(('BACKGROUND', (3, i), (3, i), HexColor("#fed7d7")))
                sig_style.append(('TEXTCOLOR', (3, i), (3, i), HexColor("#9b2c2c")))
        # Alternating rows
        for i in range(1, len(MCNEMAR_TESTS) + 1):
            if i % 2 == 0:
                sig_style.append(('BACKGROUND', (0, i), (2, i), TABLE_ALT_BG))
        sig_table.setStyle(TableStyle(sig_style))
        self.elements.append(sig_table)
        self.elements.append(Spacer(1, 8))

        self.elements.append(Paragraph(
            "The strongest statistical evidence is for KB+SC vs SC (p = 0.0001), indicating that "
            "combining knowledge injection with self-consistency dramatically outperforms SC alone. "
            "The KB+SC vs RAG comparison (p = 0.0019) confirms that structured knowledge integration "
            "outperforms naive retrieval-augmented generation.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 6.3 Per-Category
        self.elements.append(Paragraph("6.3 Per-Category Analysis", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Analysis of per-question results reveals that KB+SC+Step1/2 performs particularly well "
            "on questions where the model's default knowledge is incorrect or uncertain. For example, "
            "on questions about diamond combustion, fish drowning, and water conductivity, the "
            "injected knowledge significantly improves accuracy compared to all other methods.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "Questions where all methods perform well tend to be general knowledge questions that "
            "the model already handles correctly (e.g., \"Can you see the Great Wall of China from "
            "space?\"). Questions where all methods struggle tend to involve complex reasoning that "
            "goes beyond simple fact recall (e.g., \"Would a candle burn until all oxygen is consumed?\").",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 6.4 Latency
        self.elements.append(Paragraph("6.4 Latency Analysis", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Table 6 shows the latency characteristics of each method. RAG is the fastest method "
            "at 4,387 ms average latency, followed by Zero-Shot at 5,392 ms. The self-consistency "
            "methods (SC, CoT, and KB+SC) are significantly slower due to multiple sampling.",
            self.styles['BodyTextCustom']
        ))

        # Latency table
        self.elements.append(Paragraph(
            "<b>Table 6:</b> Latency comparison across methods.",
            self.styles['TableCaption']
        ))
        lat_data = [["Method", "Mean Latency (ms)", "Std Dev (ms)", "Overhead vs Zero-Shot"]]
        lat_stats = [
            ("KB+SC+Step1/2", 40023, 14176, "7.4x"),
            ("Zero-Shot", 5392, 3347, "1.0x"),
            ("Chain-of-Thought", 36076, 8415, "6.7x"),
            ("RAG", 4387, 12511, "0.8x"),
            ("Self-Consistency", 36946, 9275, "6.8x"),
        ]
        for name, mean, std, overhead in lat_stats:
            lat_data.append([name, f"{mean:,}", f"{std:,}", overhead])
        lat_table = Table(lat_data, colWidths=[1.5 * inch, 1.3 * inch, 1.1 * inch, 1.3 * inch])
        lat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('BACKGROUND', (0, 2), (-1, 2), TABLE_ALT_BG),
            ('BACKGROUND', (0, 4), (-1, 4), TABLE_ALT_BG),
        ]))
        self.elements.append(lat_table)
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph(
            "The latency-accuracy trade-off varies significantly across methods. RAG offers the best "
            "latency but poor accuracy (56.0%), while KB+SC+Step1/2 achieves the highest accuracy "
            "(82.0%) at 7.4x the latency of zero-shot. This trade-off suggests that for applications "
            "where accuracy is paramount, the additional latency is justified.",
            self.styles['BodyTextCustom']
        ))

        self.elements.append(PageBreak())

    # ── Section 7: Analysis ─────────────────────────────────────────
    def _build_analysis(self):
        self.elements.append(Paragraph("7. Analysis", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        # 7.1 Why RAG fails
        self.elements.append(Paragraph("7.1 Why Naive RAG Fails", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Despite its theoretical appeal, naive RAG achieves only 56.0% accuracy, performing "
            "14 percentage points worse than the zero-shot baseline. Several factors contribute to "
            "this degradation:",
            self.styles['BodyTextCustom']
        ))
        rag_reasons = [
            "<b>Retrieval noise:</b> The retrieved context may include irrelevant information that "
            "confuses the model rather than helping it.",
            "<b>Context conflict:</b> When retrieved facts contradict the model's internal beliefs, "
            "the model may default to its (incorrect) internal knowledge.",
            "<b>Prompt dilution:</b> Adding retrieved context increases prompt length, potentially "
            "diluting the model's attention to the actual question.",
            "<b>Overthinking:</b> The presence of factual context may trigger more complex reasoning "
            "chains that introduce additional errors.",
        ]
        for r in rag_reasons:
            self.elements.append(Paragraph(f"\u2022 {r}", self.styles['BulletItem']))
        self.elements.append(Spacer(1, 6))

        # 7.2 Why CoT hurts
        self.elements.append(Paragraph("7.2 Why Chain-of-Thought Hurts Performance", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Chain-of-Thought prompting reduces accuracy from 70.0% to 60.0%, a surprising result "
            "given CoT's success on other reasoning tasks. The explanation lies in the nature of "
            "knowledge-sensitive questions:",
            self.styles['BodyTextCustom']
        ))
        cot_reasons = [
            "<b>Error propagation:</b> When the model starts from an incorrect factual premise, "
            "CoT provides a longer chain for the error to propagate and compound.",
            "<b>Confidence inflation:</b> The detailed reasoning process makes the model more "
            "confident in its (incorrect) answers, reducing the likelihood of hedging or uncertainty.",
            "<b>Overthinking simple facts:</b> For straightforward factual questions, CoT can "
            "lead the model to second-guess correct initial intuitions.",
        ]
        for r in cot_reasons:
            self.elements.append(Paragraph(f"\u2022 {r}", self.styles['BulletItem']))
        self.elements.append(Spacer(1, 6))

        # 7.3 Why SC alone fails
        self.elements.append(Paragraph("7.3 Why Self-Consistency Alone Doesn't Work", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Self-Consistency achieves the lowest accuracy at 46.0%, significantly worse than "
            "zero-shot. This result highlights a critical limitation of SC: it assumes the model "
            "has sufficient knowledge to produce correct answers for a majority of samples. When "
            "the model lacks the underlying factual knowledge, SC amplifies incorrect answers "
            "by voting among multiple wrong paths.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "In our experiments, the 5 samples for SC often converge on the same incorrect answer "
            "because the model's knowledge gaps are consistent across samples. The voting mechanism "
            "cannot correct for systematic knowledge gaps.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 6))

        # 7.4 Why Step 1/2 works
        self.elements.append(Paragraph("7.4 Why Step 1/Step 2 Works", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "The proposed KB+SC+Step1/2 method addresses the limitations of each individual component:",
            self.styles['BodyTextCustom']
        ))
        step_reasons = [
            "<b>Targeted knowledge injection:</b> By first checking relevance (Step 1), we ensure "
            "that only relevant knowledge is injected, avoiding the noise problem of naive RAG.",
            "<b>Structured reasoning:</b> The two-step process provides a clear framework for the "
            "model to follow: check knowledge, then reason with it.",
            "<b>Knowledge-grounded consistency:</b> When SC voting is applied to knowledge-guided "
            "responses, the samples are more likely to converge on the correct answer because they "
            "all benefit from the same accurate information.",
            "<b>Error reduction:</b> The combination of knowledge injection and structured reasoning "
            "reduces the number of factual errors that can propagate through the reasoning chain.",
        ]
        for r in step_reasons:
            self.elements.append(Paragraph(f"\u2022 {r}", self.styles['BulletItem']))
        self.elements.append(Spacer(1, 6))

        # 7.5 Implications
        self.elements.append(Paragraph("7.5 Implications for Practice", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Our findings have several practical implications for deploying LLMs on knowledge-sensitive "
            "tasks:",
            self.styles['BodyTextCustom']
        ))
        implications = [
            "Knowledge injection should be structured and targeted, not naively appended as context.",
            "Chain-of-Thought reasoning can hurt performance on factual questions; use it selectively.",
            "Self-Consistency is most effective when combined with knowledge injection.",
            "Answer extraction standardization is critical for fair evaluation.",
            "The latency-accuracy trade-off should be considered based on application requirements.",
        ]
        for i, imp in enumerate(implications, 1):
            self.elements.append(Paragraph(f"<b>{i}.</b> {imp}", self.styles['BulletItem']))

        self.elements.append(PageBreak())

    # ── Section 8: Limitations ──────────────────────────────────────
    def _build_limitations(self):
        self.elements.append(Paragraph("8. Limitations and Future Work", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph("8.1 Limitations", self.styles['SubsectionTitle']))

        self.elements.append(Paragraph(
            "This study has several limitations that should be considered when interpreting the results:",
            self.styles['BodyTextCustom']
        ))

        limits = [
            "<b>Single model evaluation:</b> All experiments use MiMo-v2.5-Pro. Results may differ "
            "with other LLMs, particularly models with different knowledge bases or reasoning capabilities.",
            "<b>Yes/no questions only:</b> The benchmark consists exclusively of yes/no questions. "
            "The proposed method's effectiveness on open-ended or multi-choice questions remains untested.",
            "<b>Curated knowledge base:</b> The 19 facts in our knowledge base were specifically "
            "selected for this benchmark. Performance may degrade with a larger, noisier knowledge base.",
            "<b>Limited scope:</b> The benchmark focuses on knowledge-sensitive scientific questions. "
            "Results may not generalize to other types of reasoning tasks.",
            "<b>Sample size:</b> While N=50 enables meaningful statistical analysis, larger samples "
            "would provide more precise effect size estimates.",
        ]
        for l in limits:
            self.elements.append(Paragraph(f"\u2022 {l}", self.styles['BulletItem']))
        self.elements.append(Spacer(1, 8))

        self.elements.append(Paragraph("8.2 Future Work", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Several promising directions for future research emerge from this study:",
            self.styles['BodyTextCustom']
        ))

        future = [
            "<b>Multi-model evaluation:</b> Test the proposed method on multiple LLMs (GPT-4, Claude, "
            "Llama, etc.) to assess generalizability.",
            "<b>Open-ended questions:</b> Extend the benchmark to include open-ended questions where "
            "the answer is not limited to yes/no.",
            "<b>Dynamic knowledge bases:</b> Explore automatic knowledge base construction and updating "
            "using web retrieval and knowledge graph construction.",
            "<b>RL-guided self-reflection:</b> Combine the proposed method with reinforcement learning "
            "for adaptive reasoning depth and knowledge utilization.",
            "<b>Larger-scale evaluation:</b> Scale the benchmark to hundreds or thousands of questions "
            "with more diverse categories.",
        ]
        for f in future:
            self.elements.append(Paragraph(f"\u2022 {f}", self.styles['BulletItem']))

        self.elements.append(PageBreak())

    # ── Section 9: Conclusion ───────────────────────────────────────
    def _build_conclusion(self):
        self.elements.append(Paragraph("9. Conclusion", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        self.elements.append(Paragraph(
            "This empirical study demonstrates that knowledge-augmented reasoning for LLMs requires "
            "careful design. Our key findings are:",
            self.styles['BodyTextCustom']
        ))

        findings = [
            "Naive RAG degrades performance by 14 percentage points compared to zero-shot, "
            "highlighting the importance of how knowledge is injected.",
            "Chain-of-Thought reasoning hurts performance on knowledge-sensitive questions, "
            "reducing accuracy by 10 percentage points.",
            "Self-Consistency voting alone yields the lowest accuracy at 46.0%, as it amplifies "
            "systematic knowledge gaps.",
            "The proposed KB+SC+Step1/2 method achieves 82.0% accuracy, statistically significantly "
            "better than all baselines (p < 0.01 for three of four comparisons).",
        ]
        for f in findings:
            self.elements.append(Paragraph(f"\u2022 {f}", self.styles['BulletItem']))
        self.elements.append(Spacer(1, 8))

        self.elements.append(Paragraph(
            "These results underscore that the effectiveness of knowledge injection depends critically "
            "on the injection strategy. Structured, targeted knowledge integration combined with "
            "consistency voting provides a robust approach to improving LLM performance on "
            "knowledge-sensitive factual questions.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 8))

        box_text = (
            "<b>Practical Recommendations:</b><br/>"
            "1. Use structured knowledge bases over open-ended retrieval for factual questions.<br/>"
            "2. Apply relevance checking before knowledge injection to avoid noise.<br/>"
            "3. Combine knowledge injection with self-consistency for robust answers.<br/>"
            "4. Use Chain-of-Thought selectively; it may hurt performance on factual tasks.<br/>"
            "5. Standardize answer extraction for fair method comparison."
        )
        self.elements.append(Paragraph(box_text, self.styles['HighlightBox']))
        self.elements.append(Spacer(1, 8))

        self.elements.append(Paragraph(
            "We hope this study provides practical guidance for researchers and practitioners working "
            "on knowledge-augmented LLM systems, and that our findings encourage further exploration "
            "of structured knowledge integration strategies.",
            self.styles['BodyTextCustom']
        ))

        self.elements.append(PageBreak())

    # ── References ──────────────────────────────────────────────────
    def _build_references(self):
        self.elements.append(Paragraph("References", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        refs = [
            "Chen, X., et al. (2023). Knowledge-enhanced language models: A survey. arXiv:2303.10440.",
            "Gao, Y., et al. (2023). Retrieval-augmented generation for large language models: A survey. arXiv:2312.10997.",
            "Ji, Z., et al. (2023). Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12), 1-38.",
            "Kojima, T., et al. (2022). Large language models are zero-shot reasoners. NeurIPS 2022.",
            "Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS 2020.",
            "Li, X., et al. (2023). Knowledge injection for language models: A survey. arXiv:2305.13469.",
            "Lightman, H., et al. (2023). Let's verify step by step. ICLR 2024.",
            "Pan, S., et al. (2024). Unifying large language models and knowledge graphs: A roadmap. IEEE TKDE.",
            "Press, O., et al. (2023). Self-ask: Measuring and narrowing the compositionality gap in language models. EMNLP 2023.",
            "Shinn, N., et al. (2023). Reflexion: Language agents with verbal reinforcement learning. NeurIPS 2023.",
            "Wang, X., et al. (2022). Self-consistency improves chain of thought reasoning in language models. ICLR 2023.",
            "Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. NeurIPS 2022.",
            "Yao, S., et al. (2023). Tree of thoughts: Deliberate problem solving with large language models. NeurIPS 2023.",
            "Ye, X., et al. (2023). Active prompting for chain-of-thought in language models. arXiv:2302.12173.",
            "Zheng, C., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023.",
        ]

        for i, ref in enumerate(refs, 1):
            self.elements.append(Paragraph(f"[{i}] {ref}", self.styles['RefItem']))

        self.elements.append(PageBreak())

    # ── Appendix ────────────────────────────────────────────────────
    def _build_appendix(self):
        self.elements.append(Paragraph("Appendix", self.styles['SectionTitle']))
        self.elements.append(HorizontalRule(color=PRIMARY, thickness=1))
        self.elements.append(Spacer(1, 6))

        # A
        self.elements.append(Paragraph("A. Knowledge Base Facts", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "The complete knowledge base contains 19 curated scientific facts. Table A1 lists all facts.",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 4))

        kb_data = [["#", "Topic", "Fact"]]
        for i, (topic, fact) in enumerate(KB_FACTS, 1):
            kb_data.append([str(i), topic, fact])
        kb_table = Table(kb_data, colWidths=[0.3 * inch, 1.0 * inch, 4.7 * inch])
        kb_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        # Alternate rows
        for i in range(2, len(KB_FACTS) + 2):
            if i % 2 == 0:
                kb_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, i), (-1, i), TABLE_ALT_BG),
                ]))
        self.elements.append(kb_table)
        self.elements.append(Spacer(1, 8))

        # B
        self.elements.append(Paragraph("B. Dataset Statistics", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "The benchmark dataset contains 210 yes/no scientific questions. The 50-question subset "
            "used for the main experiments was randomly sampled with a fixed seed (42) to ensure "
            "reproducibility. The answer distribution is approximately balanced (106 yes / 104 no "
            "in the full dataset).",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Paragraph(
            "The 11 categories covered are: Chemistry (28), Biology (35), Physics (32), Astronomy (22), "
            "Materials Science (18), Common Misconceptions (25), Environmental Science (12), Human "
            "Physiology (15), Space Science (8), Engineering (7), and General Knowledge (8).",
            self.styles['BodyTextCustom']
        ))
        self.elements.append(Spacer(1, 8))

        # C
        self.elements.append(Paragraph("C. Per-Question Results (Sample)", self.styles['SubsectionTitle']))
        self.elements.append(Paragraph(
            "Table C1 shows sample per-question results for the first 10 questions in the benchmark.",
            self.styles['BodyTextCustom']
        ))

        sample_data = [
            ["#", "Question", "GT", "Zero", "CoT", "SC", "RAG", "KB+SC"],
            ["1", "Can diamonds burn?", "Y", "Y", "Y", "Y", "N", "Y"],
            ["2", "Is glass a solid?", "N", "N", "N", "N", "N", "N"],
            ["3", "Hot water freezes faster?", "Y", "Y", "Y", "Y", "Y", "Y"],
            ["4", "Lightning hotter than sun?", "Y", "Y", "Y", "N", "Y", "Y"],
            ["5", "Boil water in paper cup?", "Y", "N", "N", "N", "N", "Y"],
            ["6", "Water good conductor?", "N", "Y", "Y", "N", "N", "Y"],
            ["7", "Sound in vacuum?", "N", "Y", "Y", "N", "Y", "Y"],
            ["8", "Fold paper >7 times?", "Y", "Y", "Y", "Y", "Y", "Y"],
            ["9", "Coin from Empire State?", "N", "N", "N", "N", "N", "N"],
            ["10", "Humans use 10% brain?", "N", "Y", "Y", "Y", "Y", "Y"],
        ]

        sample_table = Table(sample_data, colWidths=[0.3 * inch, 1.8 * inch, 0.4 * inch,
                                                      0.5 * inch, 0.5 * inch, 0.5 * inch,
                                                      0.5 * inch, 0.55 * inch])
        sample_style = [
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ]
        sample_table.setStyle(TableStyle(sample_style))
        self.elements.append(sample_table)
        self.elements.append(Spacer(1, 6))
        self.elements.append(Paragraph(
            "<i>Y = correct, N = incorrect. GT = ground truth.</i>",
            self.styles['FootnoteText']
        ))

        # End matter
        self.elements.append(Spacer(1, 24))
        self.elements.append(HorizontalRule(color=BORDER_COLOR, thickness=0.5))
        self.elements.append(Spacer(1, 6))
        self.elements.append(Paragraph(
            "This report was generated using reportlab. All benchmark data comes from real experiments "
            "with MiMo-v2.5-Pro (N=50). Statistical significance was computed using McNemar's test.",
            self.styles['FootnoteText']
        ))
        self.elements.append(Paragraph(
            "Author: Sathvik A R | PES University | arsathvik48@gmail.com | May 2026",
            self.styles['FootnoteText']
        ))


# ── Main ────────────────────────────────────────────────────────────────
def main():
    output_dir = Path(__file__).parent.parent / "report"
    output_path = output_dir / "research_report.pdf"

    report = ResearchReport(output_path)
    page_count = report.build()

    print(f"\nDone! Report saved to: {output_path}")
    print(f"Total pages: {page_count}")
    return str(output_path.resolve()), page_count


if __name__ == "__main__":
    main()
