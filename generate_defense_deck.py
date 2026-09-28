"""
CardioAI: 7-Slide Academic & Clinical Defense Presentation Generator
Generates:
  1. 241VMTR02058_Benedict_Baah_Defense_Presentation.pptx (16:9 widescreen, custom clinical palette)
  2. 241VMTR02058_Benedict_Baah_Defense_Presentation.pdf (via PowerPoint COM automation)
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# DESIGN & COLOR PALETTE SPECIFICATIONS
# ---------------------------------------------------------------------------
# Background: Clean light off-white #F8FAFC
BG_COLOR = RGBColor(248, 250, 252)
# Card Background: Pure white #FFFFFF
CARD_BG = RGBColor(255, 255, 255)
# Deep clinical navy text/accents #0F172A
NAVY = RGBColor(15, 23, 42)
# Cool slate secondary tones #475569
SLATE = RGBColor(71, 85, 105)
# Muted slate for borders / captions #94A3B8 / #E2E8F0
LIGHT_SLATE = RGBColor(148, 163, 184)
BORDER_COLOR = RGBColor(226, 232, 240)
# Refined teal/emerald accent #0D9488
TEAL = RGBColor(13, 148, 136)
TEAL_LIGHT = RGBColor(240, 253, 250)
TEAL_BORDER = RGBColor(153, 246, 228)
# Deep navy card background for title/highlights
NAVY_CARD = RGBColor(241, 245, 249)

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"

def create_slide_base(prs):
    """Creates a blank slide with #F8FAFC background."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    
    # Background full-bleed rectangle
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
    )
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_COLOR
    bg.line.fill.background()
    return slide

def add_header(slide, tag_text, title_text, subtitle_text, slide_num):
    """Adds a standardized academic header block with slide tracker."""
    # Top Tag & Counter Container
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.2))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    # Tag line
    p_tag = tf.paragraphs[0]
    p_tag.space_after = Pt(2)
    r_tag = p_tag.add_run()
    r_tag.text = tag_text.upper()
    r_tag.font.name = FONT_HEADING
    r_tag.font.size = Pt(10)
    r_tag.font.bold = True
    r_tag.font.color.rgb = TEAL
    
    # Title
    p_title = tf.add_paragraph()
    p_title.space_after = Pt(2)
    r_title = p_title.add_run()
    r_title.text = title_text
    r_title.font.name = FONT_HEADING
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = NAVY
    
    # Subtitle
    p_sub = tf.add_paragraph()
    r_sub = p_sub.add_run()
    r_sub.text = subtitle_text
    r_sub.font.name = FONT_BODY
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = SLATE
    
    # Slide Number Badge (Top Right)
    badge = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(11.3), Inches(0.42), Inches(1.2), Inches(0.36)
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = TEAL_LIGHT
    badge.line.color.rgb = TEAL_BORDER
    badge.line.width = Pt(1)
    
    btf = badge.text_frame
    btf.margin_top = btf.margin_bottom = btf.margin_left = btf.margin_right = 0
    bp = btf.paragraphs[0]
    bp.alignment = PP_ALIGN.CENTER
    br = bp.add_run()
    br.text = f"SLIDE 0{slide_num} / 07"
    br.font.name = FONT_HEADING
    br.font.size = Pt(9.5)
    br.font.bold = True
    br.font.color.rgb = TEAL

def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_COLOR, border_width=1):
    """Draws a card shape container."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(border_width)
    else:
        card.line.fill.background()
    return card

# ---------------------------------------------------------------------------
# SLIDE BUILDERS
# ---------------------------------------------------------------------------

def build_slide_1(prs):
    """Slide 1: Title Slide (Spotlight with Balanced Whitespace)"""
    slide = create_slide_base(prs)
    
    # Top Category Badge
    pill = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.8), Inches(4.5), Inches(0.38)
    )
    pill.fill.solid()
    pill.fill.fore_color.rgb = TEAL_LIGHT
    pill.line.color.rgb = TEAL_BORDER
    pill.line.width = Pt(1)
    
    ptf = pill.text_frame
    ptf.margin_left = ptf.margin_top = ptf.margin_right = ptf.margin_bottom = 0
    pp = ptf.paragraphs[0]
    pp.alignment = PP_ALIGN.CENTER
    pr = pp.add_run()
    pr.text = "JAIN UNIVERSITY  •  MCA PROJECT DEFENSE"
    pr.font.name = FONT_HEADING
    pr.font.size = Pt(10)
    pr.font.bold = True
    pr.font.color.rgb = TEAL
    
    # Main Display Title & Subtitle Box
    t_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.733), Inches(2.2))
    tf = t_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p1 = tf.paragraphs[0]
    p1.space_after = Pt(8)
    r1 = p1.add_run()
    r1.text = "CardioAI: Predictive Modeling & Biomarker Profiling\nfor Cardiovascular Risk Stratification"
    r1.font.name = FONT_HEADING
    r1.font.size = Pt(28)
    r1.font.bold = True
    r1.font.color.rgb = NAVY
    
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = "Comparative Algorithmic Benchmarking of 21 Paradigms, Stratified 5-Fold Cross-Validation & Explainable AI (SHAP)"
    r2.font.name = FONT_BODY
    r2.font.size = Pt(14)
    r2.font.color.rgb = SLATE
    
    # Deep Teal Accent Line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(3.75), Inches(2.5), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = TEAL
    line.line.fill.background()
    
    # Metadata Grid (Two Large Clean Cards)
    # Left Card: Candidate Details
    add_card(slide, Inches(0.8), Inches(4.05), Inches(5.6), Inches(2.75))
    c_box = slide.shapes.add_textbox(Inches(1.05), Inches(4.25), Inches(5.1), Inches(2.35))
    ctf = c_box.text_frame
    ctf.word_wrap = True
    ctf.margin_left = ctf.margin_top = ctf.margin_right = ctf.margin_bottom = 0
    
    cp1 = ctf.paragraphs[0]
    cp1.space_after = Pt(2)
    cpr1 = cp1.add_run()
    cpr1.text = "RESEARCH CANDIDATE"
    cpr1.font.name = FONT_HEADING
    cpr1.font.size = Pt(10)
    cpr1.font.bold = True
    cpr1.font.color.rgb = TEAL
    
    cp2 = ctf.add_paragraph()
    cp2.space_after = Pt(4)
    cpr2 = cp2.add_run()
    cpr2.text = "Benedict Baah"
    cpr2.font.name = FONT_HEADING
    cpr2.font.size = Pt(18)
    cpr2.font.bold = True
    cpr2.font.color.rgb = NAVY
    
    items_cand = [
        ("University Seat No (USN)", "241VMTR02058"),
        ("Academic Program", "Master of Computer Applications (MCA)"),
        ("Specialization Elective", "Computer Science and IT"),
        ("Institution", "Jain University (CDOE), Bengaluru")
    ]
    for lbl, val in items_cand:
        p = ctf.add_paragraph()
        p.space_after = Pt(2)
        r_l = p.add_run()
        r_l.text = f"{lbl}: "
        r_l.font.name = FONT_BODY
        r_l.font.size = Pt(11)
        r_l.font.bold = True
        r_l.font.color.rgb = SLATE
        r_v = p.add_run()
        r_v.text = val
        r_v.font.name = FONT_BODY
        r_v.font.size = Pt(11)
        r_v.font.color.rgb = NAVY

    # Right Card: Academic Guidance & Links
    add_card(slide, Inches(6.8), Inches(4.05), Inches(5.733), Inches(2.75))
    g_box = slide.shapes.add_textbox(Inches(7.05), Inches(4.25), Inches(5.233), Inches(2.35))
    gtf = g_box.text_frame
    gtf.word_wrap = True
    gtf.margin_left = gtf.margin_top = gtf.margin_right = gtf.margin_bottom = 0
    
    gp1 = gtf.paragraphs[0]
    gp1.space_after = Pt(2)
    gpr1 = gp1.add_run()
    gpr1.text = "RESEARCH & FACULTY GUIDANCE"
    gpr1.font.name = FONT_HEADING
    gpr1.font.size = Pt(10)
    gpr1.font.bold = True
    gpr1.font.color.rgb = TEAL
    
    items_guide = [
        ("Research Supervisor & Mentor", "Dr. Kwasi Kwateng"),
        ("Academic / Faculty Instructor", "Dr. Kavitha R G"),
        ("Live Web Application", "heart-disease-prediction-...streamlit.app"),
        ("Open-Source Codebase", "github.com/BenedictBen/heart-disease-prediction"),
        ("Defense Milestone", "Semester – IV Final Viva Voce (2026)")
    ]
    for lbl, val in items_guide:
        p = gtf.add_paragraph()
        p.space_after = Pt(3)
        r_l = p.add_run()
        r_l.text = f"{lbl}: "
        r_l.font.name = FONT_BODY
        r_l.font.size = Pt(11)
        r_l.font.bold = True
        r_l.font.color.rgb = SLATE
        r_v = p.add_run()
        r_v.text = val
        r_v.font.name = FONT_BODY
        r_v.font.size = Pt(11)
        r_v.font.color.rgb = NAVY

def build_slide_2(prs):
    """Slide 2: Objectives of the Study (5 Core Research Pillars)"""
    slide = create_slide_base(prs)
    add_header(
        slide,
        tag_text="Research Foundations",
        title_text="Objectives of the Study: 5 Core Research Pillars",
        subtitle_text="Structured empirical pillars guiding clinical feature discovery, predictive modeling, and bedside delivery",
        slide_num=2
    )
    
    pillars = [
        ("01", "Biomarker Exploration", "Screen Clinical Predictors", [
            "13 physiological biomarkers analyzed.",
            "Rank hemodynamic risk significance.",
            "Identify key coronary determinants."
        ]),
        ("02", "Feature Engineering", "Optimize Physiological Signals", [
            "Leakage-free z-score scaling.",
            "Continuous median imputation.",
            "Rigorous categorical encoding."
        ]),
        ("03", "Model Architecture", "Multi-Paradigm Benchmarking", [
            "21 algorithmic architectures.",
            "Gradient boosting & baselines.",
            "Multi-tier ensemble meta-learning."
        ]),
        ("04", "Explainability (XAI)", "Transparent Risk Attribution", [
            "Patient-level SHAP values.",
            "Clinical waterfall force plots.",
            "Global feature impact rankings."
        ]),
        ("05", "Clinical Interface", "Point-of-Care CDSS", [
            "Interactive Streamlit application.",
            "Instant risk probability score.",
            "Dynamic model switching engine."
        ])
    ]
    
    card_w = Inches(2.18)
    card_gap = Inches(0.2)
    left_start = Inches(0.8)
    top_pos = Inches(1.85)
    card_h = Inches(5.1)
    
    for i, (num, title, headline, points) in enumerate(pillars):
        c_left = left_start + i * (card_w + card_gap)
        add_card(slide, c_left, top_pos, card_w, card_h)
        
        # Pill for Pillar Number
        pill = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.2), top_pos + Inches(0.25), Inches(0.75), Inches(0.35)
        )
        pill.fill.solid()
        pill.fill.fore_color.rgb = TEAL_LIGHT
        pill.line.color.rgb = TEAL_BORDER
        ptf = pill.text_frame
        ptf.margin_top = ptf.margin_bottom = ptf.margin_left = ptf.margin_right = 0
        pp = ptf.paragraphs[0]
        pp.alignment = PP_ALIGN.CENTER
        pr = pp.add_run()
        pr.text = num
        pr.font.name = FONT_HEADING
        pr.font.size = Pt(11)
        pr.font.bold = True
        pr.font.color.rgb = TEAL
        
        # Text Frame
        tb = slide.shapes.add_textbox(c_left + Inches(0.2), top_pos + Inches(0.75), card_w - Inches(0.4), card_h - Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        # Pillar Title
        p_title = tf.paragraphs[0]
        p_title.space_after = Pt(2)
        r_t = p_title.add_run()
        r_t.text = title
        r_t.font.name = FONT_HEADING
        r_t.font.size = Pt(13)
        r_t.font.bold = True
        r_t.font.color.rgb = NAVY
        
        # Headline
        p_hl = tf.add_paragraph()
        p_hl.space_after = Pt(12)
        r_hl = p_hl.add_run()
        r_hl.text = headline
        r_hl.font.name = FONT_HEADING
        r_hl.font.size = Pt(10.5)
        r_hl.font.bold = True
        r_hl.font.color.rgb = TEAL
        
        # Points (Headline + fragment rule < 8 words)
        for pt in points:
            p_pt = tf.add_paragraph()
            p_pt.space_after = Pt(8)
            r_pt = p_pt.add_run()
            r_pt.text = f"• {pt}"
            r_pt.font.name = FONT_BODY
            r_pt.font.size = Pt(10.5)
            r_pt.font.color.rgb = SLATE

def build_slide_3(prs):
    """Slide 3: Scope of the Study (Clinical Boundaries & Target Biomarkers)"""
    slide = create_slide_base(prs)
    add_header(
        slide,
        tag_text="Clinical Delimitations",
        title_text="Scope of the Study: Boundaries & Biomarker Matrix",
        subtitle_text="Defined observational patient criteria, clinical scope exclusions, and 13 target physiological features",
        slide_num=3
    )
    
    top_pos = Inches(1.85)
    col_w = Inches(5.7)
    
    # Left Column: Clinical Boundaries & Inclusions
    add_card(slide, Inches(0.8), top_pos, col_w, Inches(5.1))
    
    tb_l = slide.shapes.add_textbox(Inches(1.1), top_pos + Inches(0.25), col_w - Inches(0.6), Inches(4.6))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_top = tf_l.margin_right = tf_l.margin_bottom = 0
    
    p = tf_l.paragraphs[0]
    p.space_after = Pt(4)
    r = p.add_run()
    r.text = "CLINICAL INCLUSIONS & EXCLUSIONS"
    r.font.name = FONT_HEADING
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = NAVY
    
    p_sub = tf_l.add_paragraph()
    p_sub.space_after = Pt(14)
    r_sub = p_sub.add_run()
    r_sub.text = "Defined boundaries ensuring clinical research validity"
    r_sub.font.name = FONT_BODY
    r_sub.font.size = Pt(10.5)
    r_sub.font.color.rgb = SLATE
    
    boundaries = [
        ("Target Cohort", "Adult cardiology referral patients (Cleveland Clinic)."),
        ("Sample Size Limit", "Strict N=303 records; incomplete entries pruned."),
        ("Inclusion Protocol", "Confirmed fluoroscopy angiography and resting ECG."),
        ("Exclusion Protocol", "Pediatric cases and unverified clinical records removed."),
        ("Diagnostic Standard", "Coronary artery narrowing threshold: ≥ 50% stenosis."),
        ("Ethical Compliance", "De-identified canonical data; zero patient risk.")
    ]
    for hl, frag in boundaries:
        p = tf_l.add_paragraph()
        p.space_after = Pt(6)
        r_b = p.add_run()
        r_b.text = f"✔ {hl}: "
        r_b.font.name = FONT_HEADING
        r_b.font.size = Pt(11)
        r_b.font.bold = True
        r_b.font.color.rgb = TEAL
        
        r_f = p.add_run()
        r_f.text = frag
        r_f.font.name = FONT_BODY
        r_f.font.size = Pt(10.5)
        r_f.font.color.rgb = NAVY

    # Right Column: Target Biomarker Matrix (5 Core Features Spotlight)
    add_card(slide, Inches(6.8), top_pos, col_w, Inches(5.1))
    
    tb_r = slide.shapes.add_textbox(Inches(7.1), top_pos + Inches(0.25), col_w - Inches(0.6), Inches(4.6))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0
    
    p = tf_r.paragraphs[0]
    p.space_after = Pt(4)
    r = p.add_run()
    r.text = "TARGET BIOMARKER MATRIX (CORE 5)"
    r.font.name = FONT_HEADING
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = NAVY
    
    p_sub = tf_r.add_paragraph()
    p_sub.space_after = Pt(14)
    r_sub = p_sub.add_run()
    r_sub.text = "Key hemodynamic & fluoroscopic indicators analyzed"
    r_sub.font.name = FONT_BODY
    r_sub.font.size = Pt(10.5)
    r_sub.font.color.rgb = SLATE
    
    biomarkers = [
        ("Serum Cholesterol (chol)", "Fasting lipid profile in mg/dL; key metabolic risk marker."),
        ("Resting Blood Pressure (trestbps)", "Systolic arterial pressure in mm Hg on hospital admission."),
        ("ST Depression (oldpeak)", "Exercise-induced ST-segment myocardial ischemic strain (mm)."),
        ("Fluoroscopy Vessels (ca)", "Major coronary arteries visualized with dye blockage (0 to 3)."),
        ("Maximum Heart Rate (thalach)", "Peak exercise chronotropic capacity in beats per minute (bpm).")
    ]
    for bm, desc in biomarkers:
        p = tf_r.add_paragraph()
        p.space_after = Pt(6)
        r_b = p.add_run()
        r_b.text = f"★ {bm}\n"
        r_b.font.name = FONT_HEADING
        r_b.font.size = Pt(11)
        r_b.font.bold = True
        r_b.font.color.rgb = NAVY
        
        r_d = p.add_run()
        r_d.text = f"   {desc}"
        r_d.font.name = FONT_BODY
        r_d.font.size = Pt(10)
        r_d.font.color.rgb = SLATE

def build_slide_4(prs):
    """Slide 4: Research Methodology & Workflow Architecture"""
    slide = create_slide_base(prs)
    add_header(
        slide,
        tag_text="End-to-End Pipeline",
        title_text="Research Methodology & Workflow Architecture",
        subtitle_text="Linear 4-stage pipeline architecture from observational clinical cohort to explainable AI delivery",
        slide_num=4
    )
    
    stages = [
        ("STAGE 01", "Data Ingestion", "Raw Dataset Verification", [
            "UCI Cleveland cohort import.",
            "Verify 13 clinical biomarkers.",
            "Validate attribute data schemas.",
            "Isolate test evaluation set."
        ]),
        ("STAGE 02", "Preprocessing", "Leakage-Free Cleaning", [
            "Median imputation for continuous.",
            "Mode imputation for discrete.",
            "Pipeline-encapsulated z-scaling.",
            "Zero train-test data leakage."
        ]),
        ("STAGE 03", "Model Tuning", "Stratified Optimization", [
            "5-Fold stratified cross-validation.",
            "Systematic grid search tuning.",
            "21 algorithmic paradigms.",
            "Multi-tier stacking ensemble."
        ]),
        ("STAGE 04", "Validation & SHAP", "Clinical Explainability", [
            "ROC-AUC & Brier evaluation.",
            "500-iteration bootstrap 95% CIs.",
            "SHAP TreeExplainer attribution.",
            "Bedside waterfall risk delivery."
        ])
    ]
    
    top_pos = Inches(1.9)
    card_w = Inches(2.7)
    card_gap = Inches(0.3)
    left_start = Inches(0.8)
    card_h = Inches(5.0)
    
    for i, (badge_txt, stage_title, stage_sub, items) in enumerate(stages):
        c_left = left_start + i * (card_w + card_gap)
        add_card(slide, c_left, top_pos, card_w, card_h)
        
        # Stage Tag Pill
        pill = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.2), top_pos + Inches(0.25), Inches(1.15), Inches(0.32)
        )
        pill.fill.solid()
        pill.fill.fore_color.rgb = TEAL_LIGHT
        pill.line.color.rgb = TEAL_BORDER
        ptf = pill.text_frame
        ptf.margin_top = ptf.margin_bottom = ptf.margin_left = ptf.margin_right = 0
        pp = ptf.paragraphs[0]
        pp.alignment = PP_ALIGN.CENTER
        pr = pp.add_run()
        pr.text = badge_txt
        pr.font.name = FONT_HEADING
        pr.font.size = Pt(9.5)
        pr.font.bold = True
        pr.font.color.rgb = TEAL
        
        # Text Frame
        tb = slide.shapes.add_textbox(c_left + Inches(0.2), top_pos + Inches(0.7), card_w - Inches(0.4), card_h - Inches(0.85))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        # Stage Title
        p_t = tf.paragraphs[0]
        p_t.space_after = Pt(2)
        r_t = p_t.add_run()
        r_t.text = stage_title
        r_t.font.name = FONT_HEADING
        r_t.font.size = Pt(14)
        r_t.font.bold = True
        r_t.font.color.rgb = NAVY
        
        # Subtitle
        p_s = tf.add_paragraph()
        p_s.space_after = Pt(14)
        r_s = p_s.add_run()
        r_s.text = stage_sub
        r_s.font.name = FONT_HEADING
        r_s.font.size = Pt(10.5)
        r_s.font.bold = True
        r_s.font.color.rgb = TEAL
        
        # Checklist Items
        for it in items:
            p_i = tf.add_paragraph()
            p_i.space_after = Pt(8)
            r_i = p_i.add_run()
            r_i.text = f"• {it}"
            r_i.font.name = FONT_BODY
            r_i.font.size = Pt(10.5)
            r_i.font.color.rgb = SLATE
            
        # Directional Arrow connector between cards (if not last)
        if i < 3:
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW, c_left + card_w + Inches(0.06), top_pos + Inches(2.2), Inches(0.18), Inches(0.22)
            )
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = TEAL
            arrow.line.fill.background()

def build_slide_5(prs):
    """Slide 5: Research Design & Variable Operationalization"""
    slide = create_slide_base(prs)
    add_header(
        slide,
        tag_text="Experimental Design",
        title_text="Research Design & Variable Operationalization",
        subtitle_text="Quantitative comparative framework, clinical variable definitions, and experimental validity controls",
        slide_num=5
    )
    
    top_pos = Inches(1.85)
    card_w = Inches(3.7)
    card_gap = Inches(0.316)
    left_start = Inches(0.8)
    card_h = Inches(5.1)
    
    cards_data = [
        ("INDEPENDENT VARIABLES (X)", "Physiological Biomarkers", [
            ("Demographics", "Age (29–77 yrs), Biological Sex."),
            ("Symptom Class", "Chest pain type (Typical, Atypical, Non-anginal, Asymptomatic)."),
            ("Hemodynamics", "Resting BP, Max Heart Rate, Resting Electrocardiogram."),
            ("Ischemia Markers", "ST depression, Peak ST slope."),
            ("Fluoroscopy & Echo", "Major vessel count, Thallium scan.")
        ]),
        ("DEPENDENT VARIABLE (Y)", "Clinical Outcome Target", [
            ("Diagnostic Standard", "Coronary stenosis ≥ 50% diameter."),
            ("Binary Encoding", "0 = No CAD (54.1%), 1 = CAD Present (45.9%)."),
            ("Cohort Distribution", "164 negative controls, 139 positive cases."),
            ("Clinical Utility", "Early non-invasive triage risk score."),
            ("Target Objective", "Reduce diagnostic latency in emergency triage.")
        ]),
        ("EXPERIMENTAL CONTROLS", "Methodological Rigor", [
            ("Data Leakage Shield", "Scaling fit strictly inside train folds."),
            ("Stratified Balancing", "Identical class proportions across folds."),
            ("Baseline Controls", "Benchmarked against standard L1/L2 Logistic."),
            ("Probability Calibration", "Brier score & reliability curve analysis."),
            ("Statistical Validation", "95% bootstrap confidence intervals.")
        ])
    ]
    
    for i, (tag, title, rows) in enumerate(cards_data):
        c_left = left_start + i * (card_w + card_gap)
        add_card(slide, c_left, top_pos, card_w, card_h)
        
        tb = slide.shapes.add_textbox(c_left + Inches(0.25), top_pos + Inches(0.25), card_w - Inches(0.5), card_h - Inches(0.5))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        # Tag
        p_tag = tf.paragraphs[0]
        p_tag.space_after = Pt(2)
        r_tag = p_tag.add_run()
        r_tag.text = tag
        r_tag.font.name = FONT_HEADING
        r_tag.font.size = Pt(10)
        r_tag.font.bold = True
        r_tag.font.color.rgb = TEAL
        
        # Title
        p_t = tf.add_paragraph()
        p_t.space_after = Pt(14)
        r_t = p_t.add_run()
        r_t.text = title
        r_t.font.name = FONT_HEADING
        r_t.font.size = Pt(14)
        r_t.font.bold = True
        r_t.font.color.rgb = NAVY
        
        # Rows
        for lbl, desc in rows:
            p_r = tf.add_paragraph()
            p_r.space_after = Pt(6)
            r_l = p_r.add_run()
            r_l.text = f"• {lbl}: "
            r_l.font.name = FONT_HEADING
            r_l.font.size = Pt(10.5)
            r_l.font.bold = True
            r_l.font.color.rgb = NAVY
            
            r_d = p_r.add_run()
            r_d.text = desc
            r_d.font.name = FONT_BODY
            r_d.font.size = Pt(10)
            r_d.font.color.rgb = SLATE

def build_slide_6(prs):
    """Slide 6: Data Collection & Sampling Resampling Protocol"""
    slide = create_slide_base(prs)
    add_header(
        slide,
        tag_text="Statistical Soundness",
        title_text="Data Collection & Resampling Protocol",
        subtitle_text="Observational clinical sample curation, stratified cross-validation, and non-parametric bootstrap estimation",
        slide_num=6
    )
    
    # Top Row: 4 Big Metric Callouts
    metrics = [
        ("303 Patients", "Canonical Cleveland Cohort", "Primary observational clinical sample"),
        ("5-Fold CV", "Stratified Cross-Validation", "Variance reduction across partitions"),
        ("500 Bootstraps", "95% Confidence Intervals", "Distribution-free uncertainty estimation"),
        ("97.19% AUC", "Top Discrimination Power", "AdaBoost benchmark top performer")
    ]
    
    top_pos_m = Inches(1.85)
    card_w_m = Inches(2.7)
    card_gap_m = Inches(0.3)
    left_start = Inches(0.8)
    card_h_m = Inches(1.8)
    
    for i, (val, hl, sub) in enumerate(metrics):
        c_left = left_start + i * (card_w_m + card_gap_m)
        add_card(slide, c_left, top_pos_m, card_w_m, card_h_m, bg_color=CARD_BG, border_color=TEAL_BORDER)
        
        tb = slide.shapes.add_textbox(c_left + Inches(0.2), top_pos_m + Inches(0.18), card_w_m - Inches(0.4), card_h_m - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        # Big metric
        p_val = tf.paragraphs[0]
        p_val.space_after = Pt(2)
        r_val = p_val.add_run()
        r_val.text = val
        r_val.font.name = FONT_HEADING
        r_val.font.size = Pt(20)
        r_val.font.bold = True
        r_val.font.color.rgb = TEAL
        
        # Headline
        p_hl = tf.add_paragraph()
        p_hl.space_after = Pt(2)
        r_hl = p_hl.add_run()
        r_hl.text = hl
        r_hl.font.name = FONT_HEADING
        r_hl.font.size = Pt(11)
        r_hl.font.bold = True
        r_hl.font.color.rgb = NAVY
        
        # Subtitle
        p_sub = tf.add_paragraph()
        r_sub = p_sub.add_run()
        r_sub.text = sub
        r_sub.font.name = FONT_BODY
        r_sub.font.size = Pt(9.5)
        r_sub.font.color.rgb = SLATE

    # Bottom Row: 3 Procedural Protocol Details
    top_pos_b = Inches(3.9)
    card_w_b = Inches(3.7)
    card_gap_b = Inches(0.316)
    card_h_b = Inches(3.05)
    
    protocols = [
        ("OBSERVATIONAL COHORT", "Cleveland Clinic Foundation", [
            "Canonical clinical reference cohort.",
            "De-identified clinical catheterization records.",
            "Complete angiographic disease ground truth.",
            "Rigorous schema and data integrity verification."
        ]),
        ("STRATIFIED VALIDATION", "5-Fold Cross-Validation Shield", [
            "Maintains 46% disease prevalence in every fold.",
            "Eliminates sampling and distribution bias.",
            "Encapsulates preprocessing inside folds.",
            "Guarantees leakage-free test generalizability."
        ]),
        ("BOOTSTRAP RESAMPLING", "500-Iteration Empirical CIs", [
            "Resampling with replacement (N=303).",
            "Calculates empirical 95% Confidence Intervals.",
            "Robust against non-Gaussian distribution skew.",
            "Confirms statistically significant model gains."
        ])
    ]
    
    for i, (tag, title, points) in enumerate(protocols):
        c_left = left_start + i * (card_w_b + card_gap_b)
        add_card(slide, c_left, top_pos_b, card_w_b, card_h_b)
        
        tb = slide.shapes.add_textbox(c_left + Inches(0.25), top_pos_b + Inches(0.2), card_w_b - Inches(0.5), card_h_b - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        # Tag
        p_tag = tf.paragraphs[0]
        p_tag.space_after = Pt(2)
        r_tag = p_tag.add_run()
        r_tag.text = tag
        r_tag.font.name = FONT_HEADING
        r_tag.font.size = Pt(9.5)
        r_tag.font.bold = True
        r_tag.font.color.rgb = TEAL
        
        # Title
        p_t = tf.add_paragraph()
        p_t.space_after = Pt(8)
        r_t = p_t.add_run()
        r_t.text = title
        r_t.font.name = FONT_HEADING
        r_t.font.size = Pt(13)
        r_t.font.bold = True
        r_t.font.color.rgb = NAVY
        
        # Points
        for pt in points:
            p_p = tf.add_paragraph()
            p_p.space_after = Pt(4)
            r_p = p_p.add_run()
            r_p.text = f"• {pt}"
            r_p.font.name = FONT_BODY
            r_p.font.size = Pt(10)
            r_p.font.color.rgb = SLATE

def build_slide_7(prs):
    """Slide 7: Data Analysis Tools & Technical Environment"""
    slide = create_slide_base(prs)
    add_header(
        slide,
        tag_text="Production Infrastructure",
        title_text="Data Analysis Tools & Technical Environment",
        subtitle_text="Production-grade biomedical data science stack from runtime computation to interactive clinical delivery",
        slide_num=7
    )
    
    tiles = [
        ("RUNTIME & COMPUTE", "Core Language & Vector Math", "Python 3.11+, NumPy, Pandas", [
            "Vectorized data manipulation & transformations.",
            "Robust mathematical handling of clinical arrays.",
            "Reproducible seeds across entire experimental suite.",
            "Fast matrix calculations for high-dimensional models."
        ]),
        ("ALGORITHMIC ENGINES", "Predictive Modeling Engines", "Scikit-Learn, LightGBM, XGBoost", [
            "21 distinct classifiers systematically evaluated.",
            "Cross-validated hyperparameter optimization.",
            "Ensemble stacking with Logistic Regression meta-learner.",
            "Strict pipeline encapsulation preventing data leakage."
        ]),
        ("EXPLAINABLE AI", "Model Interpretability Framework", "SHAP (TreeExplainer & Kernel)", [
            "Shapley Additive exPlanations for all 13 features.",
            "Patient-level local risk attributions via waterfall plots.",
            "Global population beeswarm importance plots.",
            "Direct clinician auditability for bedside confidence."
        ]),
        ("DEPLOYMENT & DELIVERY", "Point-of-Care CDSS Platform", "Streamlit Cloud & Matplotlib", [
            "Zero-install responsive web browser interface.",
            "Instant risk score calculation with confidence gauge.",
            "Dynamic model switching across all 21 algorithms.",
            "Interactive unsupervised PCA & K-Means exploration."
        ])
    ]
    
    top_pos_1 = Inches(1.85)
    top_pos_2 = Inches(4.5)
    card_w = Inches(5.7)
    card_h = Inches(2.45)
    left_1 = Inches(0.8)
    left_2 = Inches(6.8)
    
    positions = [
        (left_1, top_pos_1),
        (left_2, top_pos_1),
        (left_1, top_pos_2),
        (left_2, top_pos_2)
    ]
    
    for (c_left, c_top), (tag, title, tech, points) in zip(positions, tiles):
        add_card(slide, c_left, c_top, card_w, card_h)
        
        tb = slide.shapes.add_textbox(c_left + Inches(0.25), c_top + Inches(0.18), card_w - Inches(0.5), card_h - Inches(0.35))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        # Tag
        p_tag = tf.paragraphs[0]
        p_tag.space_after = Pt(2)
        r_tag = p_tag.add_run()
        r_tag.text = tag
        r_tag.font.name = FONT_HEADING
        r_tag.font.size = Pt(9.5)
        r_tag.font.bold = True
        r_tag.font.color.rgb = TEAL
        
        # Title & Stack
        p_t = tf.add_paragraph()
        p_t.space_after = Pt(2)
        r_t = p_t.add_run()
        r_t.text = title
        r_t.font.name = FONT_HEADING
        r_t.font.size = Pt(13)
        r_t.font.bold = True
        r_t.font.color.rgb = NAVY
        
        p_tech = tf.add_paragraph()
        p_tech.space_after = Pt(6)
        r_tech = p_tech.add_run()
        r_tech.text = f"Stack: {tech}"
        r_tech.font.name = FONT_HEADING
        r_tech.font.size = Pt(10)
        r_tech.font.bold = True
        r_tech.font.color.rgb = TEAL
        
        # Points
        for pt in points:
            p_p = tf.add_paragraph()
            p_p.space_after = Pt(3)
            r_p = p_p.add_run()
            r_p.text = f"• {pt}"
            r_p.font.name = FONT_BODY
            r_p.font.size = Pt(9.8)
            r_p.font.color.rgb = SLATE

def main():
    print("Initializing 16:9 Academic Defense Presentation...")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    print("Building Slide 1: Title Slide...")
    build_slide_1(prs)
    
    print("Building Slide 2: Objectives of the Study (5 Core Pillars)...")
    build_slide_2(prs)
    
    print("Building Slide 3: Scope of the Study (Boundaries & Biomarkers)...")
    build_slide_3(prs)
    
    print("Building Slide 4: Research Methodology & Workflow Architecture...")
    build_slide_4(prs)
    
    print("Building Slide 5: Research Design & Variable Operationalization...")
    build_slide_5(prs)
    
    print("Building Slide 6: Data Collection & Resampling Protocol...")
    build_slide_6(prs)
    
    print("Building Slide 7: Data Analysis Tools & Technical Environment...")
    build_slide_7(prs)
    
    pptx_filename = "241VMTR02058_Benedict_Baah_Defense_Presentation.pptx"
    prs.save(pptx_filename)
    print(f"Successfully saved PowerPoint presentation to: {pptx_filename}")

if __name__ == "__main__":
    main()
