
# -*- coding: utf-8 -*-
import os
import docx
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register a font that supports Chinese
# Try to find a Chinese font on Windows
font_paths = [
    r"C:\Windows\Fonts\simhei.ttf",   # SimHei
    r"C:\Windows\Fonts\simsun.ttc",   # SimSun
    r"C:\Windows\Fonts\msyh.ttc",     # Microsoft YaHei
    r"C:\Windows\Fonts\simkai.ttf",   # KaiTi
]

font_registered = False
for fp in font_paths:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont('ChineseFont', fp))
            pdfmetrics.registerFont(TTFont('ChineseFont-Bold', fp))
            font_registered = True
            print(f"Using font: {fp}")
            break
        except Exception as e:
            print(f"Failed {fp}: {e}")
            continue

if not font_registered:
    raise RuntimeError("No Chinese font found!")

# Read content from docx
src = docx.Document('F:/GBGCL/2026-4-10 谭成.docx')
paragraphs_data = []
for para in src.paragraphs:
    if para.text.strip():
        paragraphs_data.append((para.style.name, para.text))

# Create PDF
output_path = 'F:/GBGCL/2026-4-10 谭成.pdf'
doc = SimpleDocTemplate(
    output_path,
    pagesize=A4,
    leftMargin=3*cm,
    rightMargin=3*cm,
    topMargin=2.5*cm,
    bottomMargin=2.5*cm
)

# Define styles
title_style = ParagraphStyle(
    'Title',
    fontName='ChineseFont-Bold',
    fontSize=18,
    leading=28,
    alignment=1,  # center
    spaceAfter=16,
    textColor=colors.HexColor('#1F2D3D')
)

h1_style = ParagraphStyle(
    'Heading1',
    fontName='ChineseFont-Bold',
    fontSize=13,
    leading=22,
    spaceBefore=14,
    spaceAfter=8,
    textColor=colors.HexColor('#1F4E79')
)

h2_style = ParagraphStyle(
    'Heading2',
    fontName='ChineseFont-Bold',
    fontSize=12,
    leading=20,
    spaceBefore=10,
    spaceAfter=6,
    textColor=colors.HexColor('#2E75B6')
)

body_style = ParagraphStyle(
    'Body',
    fontName='ChineseFont',
    fontSize=11,
    leading=20,
    spaceAfter=6,
    firstLineIndent=22
)

info_style = ParagraphStyle(
    'Info',
    fontName='ChineseFont',
    fontSize=11,
    leading=18,
    spaceAfter=4,
    textColor=colors.HexColor('#555555')
)

# Build story
story = []

for style_name, text in paragraphs_data:
    if style_name == '全文一级大标题':
        story.append(Paragraph(text, title_style))
        story.append(Spacer(1, 0.3*cm))
    elif style_name == '正文一级标题':
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(text, h1_style))
    elif style_name == '正文二级标题':
        story.append(Paragraph(text, h2_style))
    elif style_name == '正文 文本':
        # Check if it's an info line (汇报人 / 日期)
        if text.startswith('汇报人') or text.startswith('日期'):
            story.append(Paragraph(text, info_style))
        else:
            story.append(Paragraph(text, body_style))

doc.build(story)
print(f"PDF saved to: {output_path}")
