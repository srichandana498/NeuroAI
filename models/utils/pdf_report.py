"""
utils/pdf_report.py
Clinical-grade PDF report using ReportLab.
Opens inline in browser (Content-Disposition: inline).
PLACE AT: neuroai/backend/utils/pdf_report.py
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, HRFlowable)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY  = colors.HexColor("#0D1B2A")
TEAL  = colors.HexColor("#1A8C9C")
LIGHT = colors.HexColor("#F4F8FB")
ACCENT= colors.HexColor("#E8F4FD")
RED   = colors.HexColor("#E74C3C")
ORG   = colors.HexColor("#E67E22")
GRN   = colors.HexColor("#27AE60")
GRAY  = colors.HexColor("#7F8C8D")
WHITE = colors.white

def _style(name, **kw):
    defaults = dict(fontName="Helvetica", fontSize=10,
                    textColor=colors.HexColor("#2C3E50"), leading=15, spaceAfter=4)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)

H1    = _style("H1", fontName="Helvetica-Bold", fontSize=22,
               textColor=NAVY, spaceAfter=4, leading=28)
H2    = _style("H2", fontName="Helvetica-Bold", fontSize=13,
               textColor=TEAL, spaceBefore=14, spaceAfter=4, leading=18)
BODY  = _style("Body")
BOLD  = _style("Bold", fontName="Helvetica-Bold")
SMALL = _style("Small", fontSize=8, textColor=GRAY, leading=12)


def generate_report(data: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    name  = data.get("name", "—")
    dob   = data.get("dob", "—")
    adate = data.get("assessment_date", datetime.today().strftime("%d %B %Y"))
    ref   = data.get("ref_number", f"NAI-{datetime.now().strftime('%Y%m%d%H%M')}")

    hdr = Table([[
        Table([[Paragraph("NeuroAI", H1)],
               [Paragraph("Neurodevelopmental Screening Report", SMALL)]],
              colWidths=[10*cm]),
        Table([[Paragraph(f"<b>Client:</b> {name}", SMALL)],
               [Paragraph(f"<b>DOB:</b> {dob}", SMALL)],
               [Paragraph(f"<b>Date:</b> {adate}", SMALL)],
               [Paragraph(f"<b>Ref:</b> {ref}", SMALL)]],
              colWidths=[7*cm]),
    ]], colWidths=[10*cm, 7*cm])
    hdr.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")]))
    story += [hdr, HRFlowable(width="100%", thickness=2, color=TEAL),
              Spacer(1, 0.4*cm)]

    # ── Risk summary ──────────────────────────────────────────────────────────
    story.append(Paragraph("OVERALL SCREENING RESULT", H2))
    risk  = data.get("risk_level","—")
    score = data.get("risk_score", 0)
    conf  = data.get("confidence_score", 0)
    action= data.get("recommended_action","—")
    certainty = data.get("prediction_certainty","—")

    col_map = {"LOW RISK": GRN, "MODERATE RISK": ORG, "HIGH RISK": RED}
    bg_map  = {"LOW RISK": colors.HexColor("#EBF9EE"),
               "MODERATE RISK": colors.HexColor("#FEF9E7"),
               "HIGH RISK": colors.HexColor("#FDEDEC")}
    rc = col_map.get(risk, GRAY)
    rb = bg_map.get(risk, LIGHT)

    risk_tbl = Table([[
        Paragraph("<b>Risk Level</b>", BOLD),
        Paragraph(f"<b>{risk}</b>", BOLD),
        Paragraph("<b>ASD Probability</b>", BOLD),
        Paragraph(f"<b>{score:.0f}/100</b>", BOLD),
        Paragraph("<b>Confidence</b>", BOLD),
        Paragraph(f"<b>{conf:.0%}</b>", BOLD),
    ]], colWidths=[3*cm,3.5*cm,3.5*cm,2.5*cm,3*cm,2*cm])
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),rb),
        ("BOX",(0,0),(-1,-1),1.5,rc),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("TEXTCOLOR",(1,0),(1,0),rc),("TEXTCOLOR",(3,0),(3,0),rc),
    ]))
    story += [risk_tbl, Spacer(1,0.3*cm),
              Paragraph(f"<b>Recommended Action:</b> {action}", BODY),
              Paragraph(f"<b>Prediction Certainty:</b> {certainty}", BODY),
              Spacer(1, 0.5*cm)]

    # ── Domain scores ─────────────────────────────────────────────────────────
    domains = data.get("domain_scores", {})
    if domains:
        story.append(Paragraph("DOMAIN ANALYSIS", H2))
        drows = [[Paragraph("<b>Domain</b>",BOLD),
                  Paragraph("<b>Score</b>",BOLD),
                  Paragraph("<b>Level</b>",BOLD)]]
        for d, s in domains.items():
            lvl = "High" if s>=70 else ("Moderate" if s>=45 else "Low")
            drows.append([Paragraph(d,BODY),
                          Paragraph(f"{s:.0f}",BODY),
                          Paragraph(lvl,BODY)])
        dtbl = Table(drows, colWidths=[8*cm,3*cm,6*cm])
        dtbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CCCCCC")),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),6),("FONTSIZE",(0,0),(-1,-1),9),
        ]))
        story += [dtbl, Spacer(1,0.4*cm)]

        # Bar chart
        story.append(_domain_chart(domains))
        story.append(Spacer(1,0.4*cm))

    # ── Key features (SHAP) ────────────────────────────────────────────────────
    top = data.get("top_features",[])
    if top:
        story.append(Paragraph("KEY BEHAVIOURAL INDICATORS (SHAP)", H2))
        story.append(Paragraph(
            "The following features most strongly influenced the screening prediction:", BODY))
        for f in top:
            story.append(Paragraph(f"• {f.replace('_',' ').title()}", BODY))
        story.append(Spacer(1,0.3*cm))

    # ── Flags ─────────────────────────────────────────────────────────────────
    flags = data.get("behavioral_flags",[])
    if flags:
        story.append(Paragraph("BEHAVIOURAL FLAGS", H2))
        ftbl = Table([[Paragraph(f"⚠  {f}", BODY)] for f in flags],
                     colWidths=["100%"])
        ftbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#FFF9E6")),
            ("BOX",(0,0),(-1,-1),1,colors.HexColor("#F0C040")),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),10),
        ]))
        story += [ftbl, Spacer(1,0.3*cm)]

    # ── Insights ──────────────────────────────────────────────────────────────
    insights = data.get("insights",[])
    if insights:
        story.append(Paragraph("CLINICAL OBSERVATIONS", H2))
        for ins in insights:
            story.append(Paragraph(f"• {ins}", BODY))
        story.append(Spacer(1,0.3*cm))

    # ── Camera session ─────────────────────────────────────────────────────────
    sess = data.get("session_summary",{})
    if sess and sess.get("avg_attention"):
        story.append(Paragraph("CAMERA SESSION SUMMARY", H2))
        srows = [
            [Paragraph("<b>Duration</b>",BOLD),    Paragraph(f"{sess.get('duration_secs','?')}s",BODY)],
            [Paragraph("<b>Avg Attention</b>",BOLD),Paragraph(f"{sess.get('avg_attention','?')}/100",BODY)],
            [Paragraph("<b>Trend</b>",BOLD),        Paragraph(str(sess.get('attention_trend','?')).capitalize(),BODY)],
        ]
        for obs in sess.get("observations",[]):
            srows.append([Paragraph("Observation",BODY),Paragraph(obs,BODY)])
        stbl = Table(srows, colWidths=[5*cm,12*cm])
        stbl.setStyle(TableStyle([
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[ACCENT,WHITE]),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CCCCCC")),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),6),("FONTSIZE",(0,0),(-1,-1),9),
        ]))
        story += [stbl, Spacer(1,0.3*cm)]

    # ── Recommendations ────────────────────────────────────────────────────────
    recs = data.get("recommendations",[])
    if recs:
        story.append(Paragraph("CLINICAL RECOMMENDATIONS", H2))
        for i,r in enumerate(recs,1):
            story.append(Paragraph(f"{i}. {r}", BODY))
        story.append(Spacer(1,0.3*cm))

    # ── Specialist referrals (India) ───────────────────────────────────────────
    story.append(Paragraph("SPECIALIST REFERRAL OPTIONS — INDIA", H2))
    ref_rows = [
        [Paragraph("<b>Platform</b>",BOLD),Paragraph("<b>Specialty</b>",BOLD),Paragraph("<b>URL</b>",BOLD)],
        [Paragraph("Practo",BODY),       Paragraph("Developmental Pediatrics",BODY), Paragraph("practo.com/search/doctors/Developmental-Pediatrician",SMALL)],
        [Paragraph("Apollo Hospitals",BODY),Paragraph("Child Neurology / Autism",BODY),Paragraph("apollohospitals.com/find-a-doctor",SMALL)],
        [Paragraph("Lybrate",BODY),      Paragraph("Autism Specialists",BODY),       Paragraph("lybrate.com/s/autism-specialist",SMALL)],
        [Paragraph("NIMHANS",BODY),      Paragraph("Child & Adolescent Psychiatry",BODY),Paragraph("nimhans.ac.in/outpatient-services",SMALL)],
    ]
    ref_tbl = Table(ref_rows, colWidths=[3.5*cm,7*cm,6.5*cm])
    ref_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("FONTSIZE",(0,0),(-1,-1),9),
    ]))
    story += [ref_tbl, Spacer(1,0.5*cm)]

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story += [HRFlowable(width="100%",thickness=1,color=GRAY), Spacer(1,0.2*cm),
              Paragraph(
                  "DISCLAIMER: This report is a screening tool ONLY and does not constitute "
                  "a clinical diagnosis. All findings must be reviewed by a qualified healthcare "
                  "professional. This report should not replace a formal multidisciplinary assessment.",
                  SMALL)]

    doc.build(story, onFirstPage=_border, onLaterPages=_border)
    buf.seek(0)
    return buf.read()


def _domain_chart(domains):
    d   = Drawing(460, 150)
    bc  = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 60, 20, 380, 110
    bc.data = [list(domains.values())]
    bc.bars[0].fillColor = TEAL
    bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep = 0,100,20
    short = [k.split("&")[0].strip()[:14] for k in domains]
    bc.categoryAxis.categoryNames = short
    bc.categoryAxis.labels.angle = 20
    bc.categoryAxis.labels.fontSize = 7
    bc.categoryAxis.labels.dy = -5
    bc.valueAxis.labels.fontSize = 7
    d.add(bc)
    return d


def _border(canvas, doc):
    w, h = A4
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.rect(0, h-8*mm, w, 8*mm, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, 6*mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(w/2, 2*mm,
        f"Page {canvas.getPageNumber()} | NeuroAI Clinical Report | CONFIDENTIAL")
    canvas.restoreState()