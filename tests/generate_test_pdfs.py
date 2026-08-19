import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_story_comic(output_path: Path):
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        textColor=colors.HexColor('#1e1b4b'),
        spaceAfter=12
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['BodyText'],
        fontSize=12,
        leading=16,
        spaceAfter=10
    )
    
    # Page 1: Bear Story Start
    story.append(Paragraph("Chapter 1: The Bear's Forest", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "This is the story of a brown bear. The bear lives in a big green forest. "
        "The bear likes to walk around looking for sweet things. "
        "Every morning, the bear checks the hollow trees to see if there is honey.",
        body_style
    ))
    story.append(PageBreak())
    
    # Page 2: Bear Story End
    story.append(Paragraph("Chapter 1: Honey Hunt", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Today, the bear is very lucky. The bear finds a large beehive in an old oak tree. "
        "The bear climbs the tree carefully. The bear eats the delicious honey. "
        "The bear feels extremely satisfied and goes to sleep under the tree.",
        body_style
    ))
    story.append(PageBreak())
    
    # Page 3: Elephant Story
    story.append(Paragraph("Chapter 2: The Elephant's River", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "This is the story of an elephant. The elephant is a very large gray animal with a long trunk. "
        "The elephant lives in the dry savanna. Today is very hot, so the elephant is looking for water. "
        "The elephant walks toward the blue river. The elephant drinks water and sprays it over its back.",
        body_style
    ))
    
    doc.build(story)
    print(f"Generated story_comic.pdf at {output_path}")

def generate_financial_report(output_path: Path):
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        textColor=colors.HexColor('#1e1b4b'),
        spaceAfter=12
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['BodyText'],
        spaceAfter=10
    )
    
    story.append(Paragraph("Quarterly Financial Report FY 2024", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "The following table displays our GAAP and Non-GAAP operating results. "
        "Please note the strict division between GAAP and Adjusted Non-GAAP figures.",
        body_style
    ))
    
    # Table data
    data = [
        ["Year", "Revenue", "GAAP Operating Income", "Non-GAAP Operating Income", "EPS", "Adjusted EPS"],
        ["2024", "$1,000M", "$200M", "$250M", "$2.00", "$2.50"],
        ["2023", "$800M", "$150M", "$180M", "$1.50", "$1.80"]
    ]
    
    t = Table(data, colWidths=[60, 70, 120, 130, 50, 70])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e1b4b')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    
    story.append(t)
    doc.build(story)
    print(f"Generated financial_report.pdf at {output_path}")

def generate_no_evidence(output_path: Path):
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading2'],
        spaceAfter=12
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['BodyText'],
        spaceAfter=10
    )
    
    story.append(Paragraph("Gardening Tips: Roses", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "To grow beautiful roses, make sure they get at least 6 hours of direct sunlight daily. "
        "Water them deep at the roots rather than wetting the leaves. "
        "Use high-quality compost or fertilizer in early spring.",
        body_style
    ))
    
    doc.build(story)
    print(f"Generated no_evidence.pdf at {output_path}")

def main():
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    generate_story_comic(data_dir / "story_comic.pdf")
    generate_financial_report(data_dir / "financial_report.pdf")
    generate_no_evidence(data_dir / "no_evidence.pdf")

if __name__ == "__main__":
    main()
