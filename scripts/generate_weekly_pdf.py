# -*- coding: utf-8 -*-
"""Generate a styled PDF for a weekly report markdown file."""
import re
import sys
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- font setup ---
font_paths = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]
font_registered = False
for fp in font_paths:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont('C', fp))
            pdfmetrics.registerFont(TTFont('CB', fp))
            font_registered = True
            print(f"Using font: {fp}")
            break
        except Exception as e:
            print(f"Failed {fp}: {e}")
if not font_registered:
    raise RuntimeError("No Chinese font found!")

# --- styles ---
title_style = ParagraphStyle('Title', fontName='CB', fontSize=22, leading=32,
                             alignment=1, spaceAfter=4, textColor=colors.HexColor('#1F2D3D'))
subtitle_style = ParagraphStyle('Subtitle', fontName='C', fontSize=11, leading=18,
                                alignment=1, spaceAfter=10, textColor=colors.HexColor('#555555'))
callout_style = ParagraphStyle('Callout', fontName='C', fontSize=10, leading=18,
                                leftIndent=12, rightIndent=12, spaceBefore=4,
                                spaceAfter=4, backColor=colors.HexColor('#F5F7FA'),
                                borderColor=colors.HexColor('#D9E2EC'),
                                borderWidth=0.5, borderPadding=8,
                                textColor=colors.HexColor('#3D4A5C'))
h1_style = ParagraphStyle('H1', fontName='CB', fontSize=15, leading=24,
                          spaceBefore=18, spaceAfter=8, textColor=colors.HexColor('#1F4E79'))
h2_style = ParagraphStyle('H2', fontName='CB', fontSize=12.5, leading=20,
                          spaceBefore=10, spaceAfter=4, textColor=colors.HexColor('#2E75B6'))
body_style = ParagraphStyle('Body', fontName='C', fontSize=10.5, leading=20,
                            spaceAfter=4, firstLineIndent=22)
bullet_style = ParagraphStyle('Bullet', fontName='C', fontSize=10.5, leading=20,
                              leftIndent=18, bulletIndent=6, spaceAfter=2)
code_style = ParagraphStyle('Code', fontName='Courier', fontSize=9, leading=14,
                            leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=8,
                            backColor=colors.HexColor('#F0F0F0'),
                            borderColor=colors.HexColor('#D0D0D0'),
                            borderWidth=0.5, borderPadding=6)

# --- helpers ---
def md_to_reportlab(text):
    """Convert inline markdown to reportlab paragraph markup."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="Courier" color="#555">\1</font>', text)
    text = re.sub(r'\$([^$]+)\$', r'<i>\1</i>', text)
    return text

def parse_table(lines, idx):
    """Parse a markdown table starting at lines[idx]. Return (Table, new_idx)."""
    rows = []
    while idx < len(lines) and lines[idx].lstrip().startswith('|'):
        row = [c.strip() for c in lines[idx].strip().strip('|').split('|')]
        if all(set(c) <= set('-: ') for c in row):
            idx += 1
            continue
        rows.append(row)
        idx += 1
    if not rows:
        return None, idx
    header = rows[0]
    body = rows[1:]
    data = [header] + body
    col_widths = [None] * len(header)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'CB'),
        ('FONTNAME', (0, 1), (-1, -1), 'C'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8EEF5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#BCC9D8')),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t, idx

# --- main ---
src_md = sys.argv[1] if len(sys.argv) > 1 else 'F:/GBGCL/docs/weekly_reports/2026-7-10 谭成.md'
out_pdf = sys.argv[2] if len(sys.argv) > 2 else src_md.replace('.md', '.pdf')

with open(src_md, 'r', encoding='utf-8') as f:
    md = f.read()

lines = md.split('\n')

doc = SimpleDocTemplate(
    out_pdf, pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
    title='GBGCL 周报', author='谭成',
)

story = []
i = 0
in_callout = False
callout_buf = []
in_code = False
code_buf = []

def flush_callout(buf):
    if not buf:
        return
    text = ' '.join(buf).strip()
    if text:
        story.append(Paragraph(md_to_reportlab(text), callout_style))

def flush_code(buf):
    if not buf:
        return
    text = '\n'.join(buf)
    story.append(Preformatted(text, code_style))

while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    # fenced code block
    if stripped.startswith('```'):
        if in_code:
            flush_code(code_buf)
            code_buf = []
            in_code = False
        else:
            in_code = True
        i += 1
        continue
    if in_code:
        code_buf.append(line)
        i += 1
        continue

    # blockquote callout
    if stripped.startswith('>'):
        if not in_callout:
            in_callout = True
        callout_buf.append(stripped.lstrip('>').strip())
        i += 1
        continue
    else:
        if in_callout:
            flush_callout(callout_buf)
            callout_buf = []
            in_callout = False

    # title (# only at top)
    if stripped.startswith('# ') and not story:
        story.append(Paragraph(md_to_reportlab(stripped[2:]), title_style))
        i += 1
        continue

    # h1 section
    if re.match(r'^##\s+一、|^##\s+二、|^##\s+三、|^##\s+四、|^##\s+五、|^##\s+六、', stripped) or stripped.startswith('## '):
        m = re.match(r'^##\s+(.+)$', stripped)
        if m:
            story.append(Paragraph(md_to_reportlab(m.group(1)), h1_style))
        i += 1
        continue

    # h2 subsection
    if re.match(r'^###\s+\d+\.\d+', stripped):
        m = re.match(r'^###\s+(.+)$', stripped)
        if m:
            story.append(Paragraph(md_to_reportlab(m.group(1)), h2_style))
        i += 1
        continue

    # table
    if stripped.startswith('|') and i + 1 < len(lines) and lines[i+1].lstrip().startswith('|'):
        t, new_i = parse_table(lines, i)
        if t:
            story.append(t)
            story.append(Spacer(1, 0.25*cm))
            i = new_i
            continue

    # horizontal rule
    if re.match(r'^---+$', stripped):
        story.append(Spacer(1, 0.15*cm))
        i += 1
        continue

    # bullet
    if stripped.startswith('- ') or stripped.startswith('* '):
        text = stripped[2:]
        story.append(Paragraph('• ' + md_to_reportlab(text), bullet_style))
        i += 1
        continue

    # info line (汇报人 / 日期)
    if stripped.startswith('**汇报人') or stripped.startswith('汇报人'):
        story.append(Paragraph(md_to_reportlab(stripped.replace('**', '')), subtitle_style))
        i += 1
        continue

    # bold-only line as a label
    if stripped.startswith('**') and stripped.endswith('**') and len(stripped) < 60:
        story.append(Paragraph(md_to_reportlab(stripped.replace('**', '')), subtitle_style))
        i += 1
        continue

    # blank
    if not stripped:
        i += 1
        continue

    # default body paragraph
    story.append(Paragraph(md_to_reportlab(stripped), body_style))
    i += 1

# trailing callout flush
if in_callout:
    flush_callout(callout_buf)
if in_code:
    flush_code(code_buf)

doc.build(story)
print(f"PDF saved to: {out_pdf}")