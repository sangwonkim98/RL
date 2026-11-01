"""
Generate presentation .pptx file (FINAL VERSION — real 7B in-loop results)
"""
import json
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

# 결과 로드
with open("logs/final_compare.json") as f:
    R = json.load(f)
PPO = R["ppo_router"]
BM25 = R["iterative_bm25"]
d_succ = (PPO["success_rate"] - BM25["success_rate"]) * 100
d_gold = PPO["avg_gold_hit"] - BM25["avg_gold_hit"]
d_reads = PPO["avg_reads"] - BM25["avg_reads"]

# 색
C_TITLE = RGBColor(0x1A, 0x36, 0x5D)
C_TEXT = RGBColor(0x33, 0x33, 0x33)
C_ACCENT_ORANGE = RGBColor(0xE2, 0x90, 0x4A)
C_ACCENT_BLUE = RGBColor(0x45, 0x7B, 0x9D)
C_ACCENT_GREEN = RGBColor(0x2A, 0x9D, 0x8F)
C_ACCENT_RED = RGBColor(0xE6, 0x39, 0x46)
C_GRAY = RGBColor(0x77, 0x77, 0x77)
C_LIGHT_GRAY = RGBColor(0xEE, 0xEE, 0xEE)

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_textbox(slide, left, top, width, height, text, *,
                font_size=18, bold=False, color=C_TEXT, align=PP_ALIGN.LEFT,
                font_name="맑은 고딕"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]; p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size); run.font.bold = bold
    run.font.color.rgb = color; run.font.name = font_name
    return tb


def add_multiline(slide, left, top, width, height, lines, *,
                  font_size=14, color=C_TEXT, line_spacing=1.15,
                  font_name="맑은 고딕"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    for i, item in enumerate(lines):
        if isinstance(item, str):
            text, bold, c, sz = item, False, color, font_size
        else:
            text = item[0]
            bold = item[1] if len(item) > 1 else False
            c = item[2] if len(item) > 2 else color
            sz = item[3] if len(item) > 3 else font_size
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = text
        run.font.size = Pt(sz); run.font.bold = bold
        run.font.color.rgb = c; run.font.name = font_name
    return tb


def add_header(slide, title_text, subtitle=None):
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, Inches(0.08))
    bar.fill.solid(); bar.fill.fore_color.rgb = C_TITLE
    bar.line.fill.background()
    add_textbox(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.6),
                title_text, font_size=28, bold=True, color=C_TITLE)
    if subtitle:
        add_textbox(slide, Inches(0.5), Inches(0.8), Inches(12), Inches(0.4),
                    subtitle, font_size=14, color=C_GRAY)
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.25), Inches(12.33), Pt(2))
    line.fill.solid(); line.fill.fore_color.rgb = C_TITLE
    line.line.fill.background()


def add_table(slide, left, top, width, height, data, *,
              header_bold=True, font_size=12,
              col_widths=None, row_heights=None,
              header_bg=C_TITLE, header_fg=RGBColor(0xFF, 0xFF, 0xFF),
              highlight_row=None, highlight_bg=None):
    rows, cols = len(data), len(data[0])
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    if row_heights:
        for i, h in enumerate(row_heights):
            table.rows[i].height = h
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            tf = cell.text_frame
            tf.margin_left = Inches(0.08); tf.margin_right = Inches(0.08)
            tf.margin_top = Inches(0.05); tf.margin_bottom = Inches(0.05)
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(font_size); run.font.name = "맑은 고딕"
            if r == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = header_bg
                run.font.bold = header_bold
                run.font.color.rgb = header_fg
            else:
                if highlight_row is not None and r == highlight_row:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = highlight_bg or RGBColor(0xFF, 0xF2, 0xE0)
                    run.font.bold = True
                run.font.color.rgb = C_TEXT
    return table


def arrow(slide, x1, y1, x2, y2):
    line = slide.shapes.add_connector(1, x1, y1, x2, y2)
    line.line.color.rgb = C_TITLE; line.line.width = Pt(2)
    return line


def box(slide, x, y, w, h, text, fill_color, font_color=RGBColor(0xFF,0xFF,0xFF),
        font_size=14, bold=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = slide.shapes.add_shape(shape, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill_color
    sh.line.fill.background()
    tf = sh.text_frame
    tf.margin_left = Inches(0.1); tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.05); tf.margin_bottom = Inches(0.05)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size); run.font.bold = bold
    run.font.color.rgb = font_color; run.font.name = "맑은 고딕"
    return sh


def add_footer(slide, page_num, total):
    add_textbox(slide, Inches(11.8), Inches(7.1), Inches(1.4), Inches(0.3),
                f"{page_num} / {total}", font_size=10, color=C_GRAY, align=PP_ALIGN.RIGHT)


TOTAL = 14

# ─────────────────────────────────────────────────────────────────────
# 1. TITLE
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = C_TITLE
bg.line.fill.background()

add_textbox(s, Inches(1.0), Inches(2.0), Inches(11), Inches(1.2),
            "PPO Router for LLM-Guided Multi-Hop QA",
            font_size=44, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
add_textbox(s, Inches(1.0), Inches(3.3), Inches(11), Inches(0.7),
            "Reinforcement Learning over a Frozen LLM Reader",
            font_size=22, color=RGBColor(0xCC, 0xDD, 0xEE))

sep = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(4.5), Inches(2), Pt(3))
sep.fill.solid(); sep.fill.fore_color.rgb = C_ACCENT_ORANGE
sep.line.fill.background()

add_textbox(s, Inches(1.0), Inches(4.8), Inches(11), Inches(0.5),
            "HotpotQA distractor · PPO + vLLM (Qwen 7B real LLM in loop)",
            font_size=16, color=RGBColor(0xCC, 0xDD, 0xEE))
add_textbox(s, Inches(1.0), Inches(5.4), Inches(11), Inches(0.5),
            f"199 episodes  ·  100k timesteps  ·  PPO Success {PPO['success_rate']:.1%}  ·  Avg reads {PPO['avg_reads']:.2f}",
            font_size=14, color=RGBColor(0x99, 0xBB, 0xCC))

# ─────────────────────────────────────────────────────────────────────
# 2. MOTIVATION
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "Motivation", "왜 RL Router인가")

add_multiline(s, Inches(0.5), Inches(1.6), Inches(12), Inches(2.0), [
    ("기존 LLM 멀티홉 QA", True, C_TITLE, 18),
    ("• IRCoT / ReAct: LLM이 prompt 안에서 직접 '더 찾을지 / 답할지' 판단", False, C_TEXT, 15),
    ("• 멈춤 시점이 불안정 — over-reading 또는 premature stop", False, C_TEXT, 15),
    ("• 정량적 routing 신호 부재", False, C_TEXT, 15),
])

add_multiline(s, Inches(0.5), Inches(3.8), Inches(12), Inches(2.0), [
    ("우리 접근", True, C_ACCENT_ORANGE, 18),
    ("• Routing decision을 외부 RL policy(PPO)에 위임", False, C_TEXT, 15),
    ("• PPO = 의사결정 (어떤 doc 읽을지, 언제 멈출지)", False, C_TEXT, 15),
    ("• LLM = 의미 처리 (글 읽기, 답 만들기)", False, C_TEXT, 15),
    ("• 학습 환경 = 추론 환경 (LLM이 env 안에) — train=infer 일치", False, C_TEXT, 15),
])

add_footer(s, 2, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "Dataset", "HotpotQA distractor split")

add_multiline(s, Inches(0.5), Inches(1.6), Inches(6), Inches(3), [
    ("Setting", True, C_TITLE, 18),
    ("• 한 질문당 10개 문서 (gold 2 + distractor 8)", False, C_TEXT, 14),
    ("• 2-hop reasoning 필요", False, C_TEXT, 14),
    ("• 199 episodes 사용 (전체 7405중)", False, C_TEXT, 14),
    ("    bridge 166 + comparison 33", False, C_GRAY, 13),
])

add_multiline(s, Inches(7), Inches(1.6), Inches(6), Inches(3), [
    ("예시", True, C_TITLE, 18),
    ("Q: Were Scott Derrickson and Ed Wood", False, C_TEXT, 13),
    ("    of the same nationality?", False, C_TEXT, 13),
    ("A: yes", False, C_TEXT, 13),
    ("Gold: [Scott Derrickson, Ed Wood]", False, C_ACCENT_GREEN, 13),
    ("Distractor: [Doctor Strange, Sinister, ...]", False, C_GRAY, 13),
])

add_multiline(s, Inches(0.5), Inches(4.5), Inches(12), Inches(2.5), [
    ("10 docs (PPO는 인덱스/BM25점수로만 봄)", True, C_TITLE, 15),
    ("[0] Ed Wood (film) — distractor",                False, C_TEXT, 12),
    ("[1] ★ Scott Derrickson — gold",                   False, C_ACCENT_GREEN, 12),
    ("[2] Woodson, Arkansas — distractor",             False, C_TEXT, 12),
    ("[3] Tyler Bates · [4] ★ Ed Wood — gold · [5] Deliver Us · [6] Adam Collis", False, C_TEXT, 12),
    ("[7] Sinister · [8] Conrad Brooks · [9] Doctor Strange — all distractor",   False, C_TEXT, 12),
])
add_footer(s, 3, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 4. PIPELINE OVERVIEW
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "Pipeline Overview", "PPO Router + LLM in environment")

box(s, Inches(0.5), Inches(1.6), Inches(3), Inches(0.8),
    "Question + 10 docs", C_GRAY, font_size=13)
arrow(s, Inches(2), Inches(2.4), Inches(2), Inches(2.7))

box(s, Inches(0.5), Inches(2.7), Inches(3), Inches(0.7),
    "DocReaderEnv", C_TITLE, font_size=13)
arrow(s, Inches(3.5), Inches(3.05), Inches(4.5), Inches(3.05))

box(s, Inches(4.5), Inches(2.6), Inches(3.5), Inches(0.95),
    "Observation (32-dim)\nread_mask·BM25·sufficiency·step",
    C_LIGHT_GRAY, font_color=C_TEXT, font_size=11)
arrow(s, Inches(8), Inches(3.05), Inches(9), Inches(3.05))

box(s, Inches(9), Inches(2.7), Inches(3.5), Inches(0.7),
    "🟠 PPO Policy", C_ACCENT_ORANGE, font_size=14)
arrow(s, Inches(10.75), Inches(3.4), Inches(10.75), Inches(3.8))

box(s, Inches(8.5), Inches(3.8), Inches(2), Inches(0.6),
    "action 0–9 (read)", C_TITLE, font_size=11)
box(s, Inches(11), Inches(3.8), Inches(1.8), Inches(0.6),
    "action 10 (answer)", C_TITLE, font_size=11)

arrow(s, Inches(9.5), Inches(4.4), Inches(9.5), Inches(4.8))
box(s, Inches(6), Inches(4.8), Inches(7), Inches(0.85),
    "read doc_i  →  collected_text 누적  →  BM25 재계산",
    C_LIGHT_GRAY, font_color=C_TEXT, font_size=12)
arrow(s, Inches(9.5), Inches(5.65), Inches(9.5), Inches(6.05))
box(s, Inches(7), Inches(6.05), Inches(5), Inches(0.7),
    "🔵 LLM call: \"evidence sufficient? 0/1\"",
    C_ACCENT_BLUE, font_size=12)

arrow(s, Inches(11.9), Inches(4.4), Inches(11.9), Inches(4.8))
box(s, Inches(0.5), Inches(4.8), Inches(5), Inches(0.85),
    "🔵 LLM call: Q + evidence → final answer",
    C_ACCENT_BLUE, font_size=12)
arrow(s, Inches(3), Inches(5.65), Inches(3), Inches(6.05))
box(s, Inches(0.5), Inches(6.05), Inches(2.4), Inches(0.7),
    "reward +8 (correct)", C_ACCENT_GREEN, font_size=12)
box(s, Inches(3.1), Inches(6.05), Inches(2.4), Inches(0.7),
    "reward −4 (wrong)", C_ACCENT_RED, font_size=12)

add_textbox(s, Inches(0.5), Inches(6.95), Inches(12), Inches(0.3),
            "🔵 = LLM (vLLM Qwen 7B)   🟠 = PPO   회색 = env 내부", font_size=11, color=C_GRAY)

add_footer(s, 4, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 5. RL FORMULATION
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "RL Formulation", "MDP for document reading + answering")

add_textbox(s, Inches(0.5), Inches(1.5), Inches(4), Inches(0.4),
            "Action  (11 discrete)", font_size=16, bold=True, color=C_ACCENT_ORANGE)
add_multiline(s, Inches(0.5), Inches(1.95), Inches(4), Inches(3), [
    ("• read_doc_0 ~ read_doc_9", False, C_TEXT, 13),
    ("• answer", False, C_TEXT, 13),
], font_size=13)

add_textbox(s, Inches(4.7), Inches(1.5), Inches(4.3), Inches(0.4),
            "Observation  (32-dim)", font_size=16, bold=True, color=C_ACCENT_ORANGE)
add_multiline(s, Inches(4.7), Inches(1.95), Inches(4.3), Inches(3), [
    ("• read_mask  (10)", False, C_TEXT, 12),
    ("• bm25_scores  (10)", False, C_TEXT, 12),
    ("• bm25_unread_argmax  one-hot (10)", False, C_TEXT, 12),
    ("• llm_sufficient  (1)", False, C_TEXT, 12),
    ("• step_norm  (1)", False, C_TEXT, 12),
], font_size=13)

add_textbox(s, Inches(9.2), Inches(1.5), Inches(4), Inches(0.4),
            "Reward", font_size=16, bold=True, color=C_ACCENT_ORANGE)
add_multiline(s, Inches(9.2), Inches(1.95), Inches(4), Inches(3), [
    ("read gold doc      :  +1.5", False, C_ACCENT_GREEN, 12),
    ("read distractor   :  −0.4", False, C_TEXT, 12),
    ("re-read              :  −0.7", False, C_TEXT, 12),
    ("answer correct  :  +8.0", False, C_ACCENT_GREEN, 12),
    ("answer wrong    :  −4.0", False, C_ACCENT_RED, 12),
    ("step cost            :  −0.2", False, C_GRAY, 12),
], font_size=13)

sep = s.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(5), Inches(12.33), Pt(1))
sep.fill.solid(); sep.fill.fore_color.rgb = C_GRAY
sep.line.fill.background()

add_multiline(s, Inches(0.5), Inches(5.2), Inches(12), Inches(2), [
    ("Algorithm", True, C_TITLE, 16),
    ("• PPO (stable-baselines3) — MlpPolicy, 32-dim obs → softmax(11)", False, C_TEXT, 13),
    ("• n_steps=1024, batch=128, n_epochs=8, lr=5e-4, ent_coef=0.03, γ=0.99", False, C_TEXT, 13),
    ("• max_steps=4 per episode (강제 종료)", False, C_TEXT, 13),
    ("• real Qwen 7B in env loop, 100k timesteps (~53min)", False, C_TEXT, 13),
])

add_footer(s, 5, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 6. LLM CALLS
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "LLM Calls", "두 종류만 — 환경 안 LLM 인터페이스")

add_textbox(s, Inches(0.5), Inches(1.5), Inches(6), Inches(0.4),
            "🔵 Call 1: Sufficiency check  (매 read 후)",
            font_size=16, bold=True, color=C_ACCENT_BLUE)

box1 = s.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(2.0), Inches(6.2), Inches(3.8))
box1.fill.solid(); box1.fill.fore_color.rgb = RGBColor(0xF5, 0xF8, 0xFC)
box1.line.color.rgb = C_ACCENT_BLUE; box1.line.width = Pt(1)
add_multiline(s, Inches(0.7), Inches(2.1), Inches(6), Inches(3.6), [
    ("Question: Were Scott Derrickson and Ed Wood", False, C_TEXT, 11),
    ("of the same nationality?", False, C_TEXT, 11),
    ("", False, C_TEXT, 11),
    ("Evidence:", False, C_TEXT, 11),
    ("[Doc 1] Scott Derrickson: ... American director ...", False, C_TEXT, 11),
    ("[Doc 4] Ed Wood: ... American filmmaker ...", False, C_TEXT, 11),
    ("", False, C_TEXT, 11),
    ("Is the evidence sufficient? Respond with one", False, C_TEXT, 11),
    ("digit: 0 (insufficient) or 1 (sufficient).", False, C_TEXT, 11),
], font_size=11)
add_textbox(s, Inches(0.5), Inches(5.95), Inches(6), Inches(0.4),
            "max_tokens=2, temp=0.0  → \"1\"", font_size=12, color=C_GRAY)

add_textbox(s, Inches(6.95), Inches(1.5), Inches(6), Inches(0.4),
            "🔵 Call 2: Answer generation  (action=10)",
            font_size=16, bold=True, color=C_ACCENT_BLUE)

box2 = s.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, Inches(7), Inches(2.0), Inches(6.2), Inches(3.8))
box2.fill.solid(); box2.fill.fore_color.rgb = RGBColor(0xF5, 0xF8, 0xFC)
box2.line.color.rgb = C_ACCENT_BLUE; box2.line.width = Pt(1)
add_multiline(s, Inches(7.2), Inches(2.1), Inches(6), Inches(3.6), [
    ("Question: Were Scott Derrickson and Ed Wood", False, C_TEXT, 11),
    ("of the same nationality?", False, C_TEXT, 11),
    ("", False, C_TEXT, 11),
    ("Evidence:", False, C_TEXT, 11),
    ("[Doc 1] Scott Derrickson: ...", False, C_TEXT, 11),
    ("[Doc 4] Ed Wood: ...", False, C_TEXT, 11),
    ("", False, C_TEXT, 11),
    ("Provide ONLY the short final answer.", False, C_TEXT, 11),
    ("Answer:", False, C_TEXT, 11),
], font_size=11)
add_textbox(s, Inches(7), Inches(5.95), Inches(6), Inches(0.4),
            "max_tokens=32  → \"Yes\"  → gold=\"yes\" 매칭 → +8",
            font_size=12, color=C_GRAY)

add_multiline(s, Inches(0.5), Inches(6.5), Inches(12.5), Inches(0.6), [
    ("LLM cache: (episode_id, read_mask) → output. 같은 조합 두 번째부터 hit (~0초). "
     "한 episode 평균 LLM 호출 3–4회.", False, C_GRAY, 12),
])

add_footer(s, 6, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 7. RUNNING EXAMPLE
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "Running Example", "한 episode 끝까지 따라가기 (Scott Derrickson example)")

qbox = s.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.4), Inches(12.33), Inches(0.7))
qbox.fill.solid(); qbox.fill.fore_color.rgb = RGBColor(0xF0, 0xF4, 0xF8)
qbox.line.color.rgb = C_TITLE; qbox.line.width = Pt(1)
add_multiline(s, Inches(0.7), Inches(1.45), Inches(12), Inches(0.6), [
    ("Q: Were Scott Derrickson and Ed Wood of the same nationality?", True, C_TITLE, 13),
    ("gold_indices=[1, 4]   ·   gold_answer=\"yes\"   ·   max_steps=4", False, C_GRAY, 11),
])

col_w = Inches(4.05); col_h = Inches(4.4); y0 = Inches(2.3)
step_xs = [Inches(0.5), Inches(4.65), Inches(8.8)]
step_titles = ["Step 1", "Step 2", "Step 3 (final)"]
step_colors = [C_TITLE, C_TITLE, C_ACCENT_GREEN]

for x, title, c in zip(step_xs, step_titles, step_colors):
    cb = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y0, col_w, col_h)
    cb.fill.solid(); cb.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    cb.line.color.rgb = c; cb.line.width = Pt(1.5)
    title_box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y0, col_w, Inches(0.35))
    title_box.fill.solid(); title_box.fill.fore_color.rgb = c
    title_box.line.fill.background()
    add_textbox(s, x, y0 + Emu(20000), col_w, Inches(0.3),
                title, font_size=13, bold=True,
                color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)

add_multiline(s, Inches(0.65), Inches(2.75), Inches(3.85), Inches(4), [
    ("state", True, C_GRAY, 9),
    ("read_mask=[0,0,0,0,0,0,0,0,0,0]", False, C_TEXT, 9),
    ("llm_sufficient=0   step=0/4", False, C_TEXT, 9),
    ("BM25 argmax(unread)=doc#1", False, C_TEXT, 9),
    ("", False, C_TEXT, 7),
    ("🟠 PPO action", True, C_ACCENT_ORANGE, 10),
    ("read doc#1 ★ Scott Derrickson", False, C_TEXT, 10),
    ("", False, C_TEXT, 6),
    ("🔵 LLM sufficiency call", True, C_ACCENT_BLUE, 10),
    ("prompt: Q + [Doc 1] Scott…American director…", False, C_TEXT, 8),
    ("           Is the evidence sufficient? (0/1)", False, C_TEXT, 8),
    ("response: \"0\"", False, C_ACCENT_BLUE, 10),
    ("", False, C_TEXT, 6),
    ("reward", True, C_GRAY, 9),
    ("+1.5 (gold) −0.2 (step) = +1.3", False, C_ACCENT_GREEN, 10),
], line_spacing=1.0)

add_multiline(s, Inches(4.8), Inches(2.75), Inches(3.85), Inches(4), [
    ("state", True, C_GRAY, 9),
    ("read_mask=[0,1,0,0,0,0,0,0,0,0]", False, C_TEXT, 9),
    ("llm_sufficient=0   step=1/4", False, C_TEXT, 9),
    ("BM25 재계산 → doc#4 위로", False, C_TEXT, 9),
    ("", False, C_TEXT, 7),
    ("🟠 PPO action", True, C_ACCENT_ORANGE, 10),
    ("read doc#4 ★ Ed Wood", False, C_TEXT, 10),
    ("", False, C_TEXT, 6),
    ("🔵 LLM sufficiency call", True, C_ACCENT_BLUE, 10),
    ("prompt: Q + [Doc 1]… + [Doc 4] Ed Wood…American filmmaker…", False, C_TEXT, 8),
    ("           Is the evidence sufficient? (0/1)", False, C_TEXT, 8),
    ("response: \"1\"   ← suff signal 바뀜!", False, C_ACCENT_BLUE, 10),
    ("", False, C_TEXT, 6),
    ("reward", True, C_GRAY, 9),
    ("+1.5 (gold) −0.2 (step) = +1.3", False, C_ACCENT_GREEN, 10),
], line_spacing=1.0)

add_multiline(s, Inches(8.95), Inches(2.75), Inches(3.85), Inches(4), [
    ("state", True, C_GRAY, 9),
    ("read_mask=[0,1,0,0,1,0,0,0,0,0]", False, C_TEXT, 9),
    ("llm_sufficient=1   step=2/4", False, C_ACCENT_GREEN, 9),
    ("(PPO가 신호 받음)", False, C_GRAY, 9),
    ("", False, C_TEXT, 7),
    ("🟠 PPO action", True, C_ACCENT_ORANGE, 10),
    ("ANSWER  (action=10)", False, C_TEXT, 10),
    ("", False, C_TEXT, 6),
    ("🔵 LLM answer call", True, C_ACCENT_BLUE, 10),
    ("prompt: Q + [Doc 1]… + [Doc 4]…", False, C_TEXT, 8),
    ("           Provide ONLY the short answer.", False, C_TEXT, 8),
    ("response: \"Yes\"   →  gold=\"yes\" ✓", False, C_ACCENT_GREEN, 10),
    ("", False, C_TEXT, 6),
    ("reward", True, C_GRAY, 9),
    ("+8.0 (correct) −0.2 (step) = +7.8", False, C_ACCENT_GREEN, 10),
], line_spacing=1.0)

for i in range(2):
    arrow(s, step_xs[i] + col_w, Inches(4.5), step_xs[i+1], Inches(4.5))

result_box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 Inches(0.5), Inches(6.85), Inches(12.33), Inches(0.5))
result_box.fill.solid(); result_box.fill.fore_color.rgb = C_ACCENT_GREEN
result_box.line.fill.background()
add_textbox(s, Inches(0.5), Inches(6.9), Inches(12.33), Inches(0.4),
            "Total return = 1.3 + 1.3 + 7.8 = +10.4   ✓ Success    "
            "(LLM 호출 총 3회: sufficiency × 2 + answer × 1)",
            font_size=12, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)

add_footer(s, 7, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 8. RESULTS — 깔끔한 2-method 비교
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "Results", f"199 episodes, real 7B in env (학습/평가 동일 환경)")

table_data = [
    ["Method", "Success Rate", "Avg Gold Hit", "Avg Reads"],
    ["Iterative BM25 (greedy, no RL)",
     f"{BM25['success_rate']:.1%}",
     f"{BM25['avg_gold_hit']:.2f} / 2",
     f"{BM25['avg_reads']:.2f}"],
    ["PPO Router (ours, 100k)",
     f"{PPO['success_rate']:.1%}",
     f"{PPO['avg_gold_hit']:.2f} / 2",
     f"{PPO['avg_reads']:.2f}"],
]
add_table(s, Inches(1.5), Inches(1.7), Inches(10), Inches(2.0),
          table_data,
          col_widths=[Inches(4.0), Inches(2.0), Inches(2.0), Inches(2.0)],
          font_size=14, highlight_row=2)

add_multiline(s, Inches(0.5), Inches(4.5), Inches(12.5), Inches(2.7), [
    ("핵심 결과", True, C_TITLE, 18),
    (f"• PPO Success {PPO['success_rate']:.1%}  vs  Iterative BM25 {BM25['success_rate']:.1%}  →  Δ {d_succ:+.1f}%p",
     True, C_ACCENT_GREEN, 14),
    (f"• PPO Reads {PPO['avg_reads']:.2f}  vs  BM25 {BM25['avg_reads']:.2f}  →  더 적게 읽음 ({d_reads:+.2f})",
     True, C_ACCENT_GREEN, 14),
    (f"• Gold hit은 비슷 ({PPO['avg_gold_hit']:.2f} vs {BM25['avg_gold_hit']:.2f})  →  doc 선별 능력은 비슷",
     False, C_TEXT, 14),
    ("", False, C_TEXT, 8),
    ("→  PPO는 단순히 더 좋은 doc을 고르는 게 아니라, ", False, C_TEXT, 14),
    ("    \"언제 멈춰 답할지\"를 학습 — 더 적은 read로 더 높은 success", True, C_TITLE, 14),
])

add_footer(s, 8, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 9. TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "Training Curves", "PPO real 7B in loop, 100k timesteps (~53min)")

s.shapes.add_picture("logs/training_curves_real_7b.png",
                      Inches(0.5), Inches(1.6), Inches(12.3), Inches(3.5))

add_multiline(s, Inches(0.5), Inches(5.3), Inches(12.5), Inches(2), [
    ("• ep_rew_mean: −1.70 → +5.25  (학습 진행)", False, C_TEXT, 14),
    ("• ep_len_mean: 3.43 → 3.12  (점점 일찍 답 — read 횟수 감소 학습)", False, C_TEXT, 14),
    ("• entropy 감소 → policy가 deterministic 해짐", False, C_TEXT, 14),
])

add_footer(s, 9, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 10. DEMO SUCCESS
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "Demo Trace — Success", "idx=5  ·  '2014 S/S debut album'")

add_textbox(s, Inches(0.5), Inches(1.5), Inches(12.5), Inches(0.4),
            "Q: 2014 S/S is the debut album of a South Korean boy group that was formed by who?",
            font_size=14, bold=True, color=C_TITLE)
add_textbox(s, Inches(0.5), Inches(1.95), Inches(12.5), Inches(0.4),
            "Gold answer: YG Entertainment   ·   Gold docs: [2014 S/S, Winner (band)]",
            font_size=13, color=C_GRAY)

trace_lines = [
    ("[Step 1]  read_mask=[0,0,0,0,0,0,0,0,0,0]  BM25 top=doc#2", False, C_TEXT, 12),
    ("           PPO → read doc#2 ★ [2014 S/S]   r=+1.30", False, C_ACCENT_GREEN, 12),
    ("           LLM sufficient: 0", False, C_GRAY, 11),
    ("", False, C_TEXT, 11),
    ("[Step 2]  read_mask=[0,0,1,0,0,0,0,0,0,0]  BM25 top=doc#4", False, C_TEXT, 12),
    ("           PPO → read doc#4 ★ [Winner (band)]   r=+1.30", False, C_ACCENT_GREEN, 12),
    ("           LLM sufficient: 1   ← 신호 받음", False, C_ACCENT_BLUE, 11),
    ("", False, C_TEXT, 11),
    ("[Step 3]  PPO → ANSWER (action=10)", False, C_TEXT, 12),
    ("           🔵 LLM: 'YG Entertainment'   →  gold='YG Entertainment'   ✓", False, C_ACCENT_GREEN, 12),
    ("           reward = +7.80    total_return = +10.40    success ✓", False, C_ACCENT_GREEN, 12),
]
add_multiline(s, Inches(0.5), Inches(2.5), Inches(12.5), Inches(4.5),
              trace_lines, font_size=13)

add_footer(s, 10, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 11. DEMO FAILURE
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "Demo Trace — Failure", "BM25 fooled by distractor")

add_textbox(s, Inches(0.5), Inches(1.5), Inches(12.5), Inches(0.4),
            "Q: Were Scott Derrickson and Ed Wood of the same nationality?",
            font_size=14, bold=True, color=C_TITLE)
add_textbox(s, Inches(0.5), Inches(1.95), Inches(12.5), Inches(0.4),
            "Gold docs: [Scott Derrickson, Ed Wood]   ·   Gold answer: yes",
            font_size=13, color=C_GRAY)

trace_lines = [
    ("[Step 1]  BM25 top-1 = doc#9 [Doctor Strange (2016 film)] ← distractor!", False, C_ACCENT_RED, 12),
    ("           Scott Derrickson이 감독한 영화라 BM25 점수 최고", False, C_GRAY, 11),
    ("           PPO → read doc#9   r=−0.60   (distractor)", False, C_ACCENT_RED, 12),
    ("", False, C_TEXT, 11),
    ("[Step 2]  PPO → read doc#7 [Sinister (film)]   r=−0.60", False, C_ACCENT_RED, 12),
    ("[Step 3]  PPO → read doc#5 [Deliver Us from Evil]   r=−0.60", False, C_ACCENT_RED, 12),
    ("[Step 4]  PPO → read doc#1 ★ [Scott Derrickson]   r=+1.30", False, C_ACCENT_GREEN, 12),
    ("           max_steps 도달, doc#4 [Ed Wood] 못 읽음, answer 못 침", False, C_ACCENT_RED, 12),
    ("           total_return = −0.50    success ✗", False, C_ACCENT_RED, 12),
    ("", False, C_TEXT, 11),
    ("Lesson: BM25가 distractor를 더 높이 점수 주는 hard case에서 PPO도 실패", False, C_TITLE, 13),
]
add_multiline(s, Inches(0.5), Inches(2.5), Inches(12.5), Inches(4.5),
              trace_lines, font_size=13)

add_footer(s, 11, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 12. WHAT DID PPO ACTUALLY LEARN
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "What Did PPO Actually Learn", "동일한 BM25 신호 → 다른 행동")

add_multiline(s, Inches(0.5), Inches(1.6), Inches(12.5), Inches(5), [
    ("관찰 1: Gold hit은 비슷", True, C_TITLE, 16),
    (f"  • PPO: {PPO['avg_gold_hit']:.2f}/2   ·   Iterative BM25: {BM25['avg_gold_hit']:.2f}/2", False, C_TEXT, 13),
    ("  → 둘 다 BM25 기반이라 'gold doc 찾기' 능력은 비슷", False, C_GRAY, 13),
    ("", False, C_TEXT, 10),
    ("관찰 2: 그런데 success는 PPO가 2배", True, C_TITLE, 16),
    (f"  • PPO: {PPO['success_rate']:.1%}   ·   Iterative BM25: {BM25['success_rate']:.1%}", False, C_TEXT, 13),
    ("", False, C_TEXT, 10),
    ("관찰 3: PPO는 더 적게 읽음", True, C_TITLE, 16),
    (f"  • PPO: {PPO['avg_reads']:.2f} reads   ·   Iterative BM25: {BM25['avg_reads']:.2f} reads", False, C_TEXT, 13),
    ("", False, C_TEXT, 10),
    ("→ PPO가 학습한 것: \"언제 멈춰 답할지\"", True, C_ACCENT_ORANGE, 17),
    ("    • Iterative BM25는 LLM suff=1 신호 그대로 따름 (수동)", False, C_TEXT, 13),
    ("    • PPO는 read_mask, BM25 점수 추세, suff 신호를 종합해 능동적으로 멈춤", False, C_TEXT, 13),
    ("    • 불필요한 distractor 읽기를 회피 → LLM이 evidence noise 없이 답 생성", False, C_TEXT, 13),
])

add_footer(s, 12, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 13. LIMITATIONS
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "Limitations", "솔직한 평가")

add_multiline(s, Inches(0.5), Inches(1.6), Inches(12.5), Inches(5.5), [
    ("1. 데이터 규모 작음", True, C_ACCENT_RED, 16),
    ("    • 199 episodes만 사용 (전체 7405중)", False, C_TEXT, 13),
    ("    • 더 큰 데이터에서의 일반화는 미검증", False, C_TEXT, 13),
    ("", False, C_TEXT, 10),
    ("2. 2-hop으로 한정", True, C_ACCENT_RED, 16),
    ("    • bridge / comparison 모두 2-hop. 3-hop+ 일반화 미확인", False, C_TEXT, 13),
    ("", False, C_TEXT, 10),
    ("3. Feature engineering 의존", True, C_ACCENT_RED, 16),
    ("    • bm25_unread_argmax 같은 strong feature가 학습에 결정적 (별도 ablation)", False, C_TEXT, 13),
    ("    • PPO가 raw text 위에서 처음부터 학습한 것은 아님", False, C_TEXT, 13),
    ("", False, C_TEXT, 10),
    ("4. BM25가 잘못 점수 주는 hard case에 약함", True, C_ACCENT_RED, 16),
    ("    • Distractor가 BM25 top 차지하면 PPO도 fooled (Demo failure case 참고)", False, C_TEXT, 13),
    ("", False, C_TEXT, 10),
    ("5. 평가 데이터 = 학습 데이터", True, C_ACCENT_RED, 16),
    ("    • 같은 199 episodes에서 학습 + 평가 (held-out test split 미수행)", False, C_TEXT, 13),
])

add_footer(s, 13, TOTAL)

# ─────────────────────────────────────────────────────────────────────
# 14. SUMMARY + FUTURE WORK
# ─────────────────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
add_header(s, "Summary & Future Work", "")

add_multiline(s, Inches(0.5), Inches(1.6), Inches(12.5), Inches(3), [
    ("Summary", True, C_TITLE, 18),
    ("• PPO Router + LLM-in-env 파이프라인 — 학습/추론 환경 일치 (real 7B)", False, C_TEXT, 14),
    (f"• PPO Success {PPO['success_rate']:.1%}  vs  Iterative BM25 {BM25['success_rate']:.1%}  ({d_succ:+.1f}%p)",
     False, C_ACCENT_GREEN, 14),
    (f"• PPO Avg reads {PPO['avg_reads']:.2f}  vs  BM25 {BM25['avg_reads']:.2f}  ({d_reads:+.2f}, 더 적게 읽음)",
     False, C_ACCENT_GREEN, 14),
    ("• 같은 BM25 신호 + 같은 LLM 위에서, RL이 \"언제 멈출지\"를 학습 — efficiency gain 확인", False, C_TEXT, 14),
])

add_multiline(s, Inches(0.5), Inches(4.7), Inches(12.5), Inches(2.5), [
    ("Future Work", True, C_ACCENT_ORANGE, 18),
    ("• Held-out test split — train/eval 분리 후 일반화 검증", False, C_TEXT, 14),
    ("• 3-hop+ 데이터 (MuSiQue 등) — 더 깊은 multi-hop reasoning", False, C_TEXT, 14),
    ("• Full wiki retrieval — BM25가 약한 환경에서의 PPO 가치 검증", False, C_TEXT, 14),
    ("• Hard-case 보강 — distractor가 BM25 top 차지하는 episode에서 robust한 정책", False, C_TEXT, 14),
])

add_footer(s, 14, TOTAL)

# ─────────────────────────────────────────────────────────────────────
out = "PPO_Router_Presentation.pptx"
prs.save(out)
print(f"Saved: {out}")
print(f"Total slides: {len(prs.slides)}")
print()
print("=" * 60)
print("Key numbers used:")
print(f"  PPO  : success={PPO['success_rate']:.2%}  gold_hit={PPO['avg_gold_hit']:.2f}  reads={PPO['avg_reads']:.2f}")
print(f"  BM25 : success={BM25['success_rate']:.2%}  gold_hit={BM25['avg_gold_hit']:.2f}  reads={BM25['avg_reads']:.2f}")
print(f"  Δ    : succ={d_succ:+.1f}%p  gold={d_gold:+.2f}  reads={d_reads:+.2f}")
