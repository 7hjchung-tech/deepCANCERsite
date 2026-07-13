# -*- coding: utf-8 -*-
"""explain_blockb.py — 조원 설명용 Block B 그림 PDF 생성 (reportlab 벡터 그래픽).

  python explain_blockb.py test   # 다이어그램만 한 장에 모아 _diagtest.pdf
  python explain_blockb.py        # 전체 문서 Block_B_설명.pdf
"""

import sys, json, math
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, HRFlowable)
from reportlab.lib.styles import ParagraphStyle
from reportlab.graphics.shapes import Drawing, Line, Circle, Rect, String, PolyLine, Polygon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
SANS = "HYGothic-Medium"

INK = HexColor("#1a1a2e"); TEAL = HexColor("#0f6d6d"); RED = HexColor("#c0392b")
GREEN = HexColor("#1e8449"); ORANGE = HexColor("#e07b00"); BLUE = HexColor("#2563eb")
GRAY = HexColor("#8a97a0"); LGRAY = HexColor("#cfd8d8"); PANEL = HexColor("#eef4f4")

R = json.load(open("analysis_results.json"))
CD = {n: v for n, v in R["corr_del"]}
CW = 460   # content width


def S(g, x, y, t, size=8, color=INK, anchor="middle", bold=False):
    s = String(x, y, t); s.fontName = SANS; s.fontSize = size
    s.fillColor = color; s.textAnchor = anchor
    g.add(s)


def bead(g, cx, cy, r=6, fill=TEAL, ghost=False):
    c = Circle(cx, cy, r)
    if ghost:
        c.fillColor = None; c.strokeColor = GRAY; c.strokeWidth = 1.1; c.strokeDashArray = [2, 2]
    else:
        c.fillColor = fill; c.strokeColor = white; c.strokeWidth = 1.2
    g.add(c)


def arrow(g, x1, y1, x2, y2, color, w=2.2, head=6):
    g.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=w))
    ang = math.atan2(y2 - y1, x2 - x1)
    for da in (math.radians(148), math.radians(-148)):
        g.add(Line(x2, y2, x2 + head * math.cos(ang + da), y2 + head * math.sin(ang + da),
                   strokeColor=color, strokeWidth=w))


def arc(x0, y0, x1, y1, peak, n=22):
    pts = []
    for i in range(n + 1):
        t = i / n
        pts += [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t + peak * 4 * t * (1 - t)]
    return pts


# ===== Diagram A : SAV / deletion / insertion =====
def diag_chain_edits():
    d = Drawing(CW, 195)
    def label(y, num, sub):
        S(d, 8, y + 2, num, 9.5, TEAL, "start"); S(d, 8, y - 11, sub, 7.5, GRAY, "start")
    x0, sp = 112, 16
    xs = [x0 + i * sp for i in range(9)]
    # ① SAV
    y = 158; label(y, "① SAV", "치환")
    for i, x in enumerate(xs):
        bead(d, x, y, fill=(RED if i == 4 else TEAL))
    S(d, 268, y, "길이/위치 그대로 → 점 하나", 8, INK, "start")
    # ② deletion
    y = 98; label(y, "② 결실", "잘라내고 봉합")
    for i, x in enumerate(xs):
        bead(d, x, y, ghost=(i in (4, 5, 6)))
    g = d
    g.add(PolyLine(arc(xs[3], y - 2, xs[7], y - 2, -26), strokeColor=RED, strokeWidth=2, strokeDashArray=[3, 2]))
    S(d, 268, y, "토막 제거 → 남은 양끝 다시 이음", 8, INK, "start")
    # ③ insertion
    y = 38; label(y, "③ 삽입", "끼워넣어 길어짐")
    xins = [x0 + i * sp for i in range(5)] + [x0 + 5 * sp + 6, x0 + 6 * sp + 6] + [x0 + 7 * sp + 12, x0 + 8 * sp + 12]
    for i, x in enumerate(xins):
        bead(d, x, y, fill=(ORANGE if i in (5, 6) else TEAL))
    S(d, 300, y, "잔기 삽입 → 길이 증가", 8, INK, "start")
    return d


# ===== Diagram B : junction gap (surface vs core deletion) =====
def diag_junction():
    d = Drawing(CW, 200)
    def panel(x0, title, e1x, e2x, col, cap, badge):
        d.add(Rect(x0, 18, 220, 170, fillColor=PANEL, strokeColor=LGRAY, strokeWidth=1, rx=8, ry=8))
        S(d, x0 + 12, 172, title, 10, TEAL, "start")
        ey = 104
        d.add(PolyLine(arc(e1x, ey, e2x, ey, 52), strokeColor=GRAY, strokeWidth=1.8, strokeDashArray=[4, 3]))
        S(d, (e1x + e2x) / 2, ey + 60, "삭제 잔기", 7.5, GRAY, "middle")
        d.add(Line(e1x + 6, ey, e2x - 6, ey, strokeColor=col, strokeWidth=2.6))
        bead(d, e1x, ey, r=6, fill=BLUE); bead(d, e2x, ey, r=6, fill=BLUE)
        S(d, e1x, ey - 16, "p-1", 7.5, INK); S(d, e2x, ey - 16, "q+1", 7.5, INK)
        S(d, (e1x + e2x) / 2, 62, cap, 8, col, "middle")
        d.add(Rect(x0 + 20, 28, 180, 22, fillColor=col, strokeColor=None, rx=6, ry=6))
        S(d, x0 + 110, 40, badge, 9, white, "middle")
    panel(5, "표면 loop 결실", 66, 126, GREEN, "gap 작음", "봉합 쉬움 → 관대")
    panel(235, "Core β-strand 결실", 280, 420, RED, "gap 큼", "봉합 불가 → 치명적")
    return d


# ===== Diagram C : pLDDT as flexibility axis (RAD51C chain) =====
def diag_plddt():
    d = Drawing(CW, 116)
    bar_y, bar_h = 60, 26
    d.add(Rect(40, bar_y, 30, bar_h, fillColor=ORANGE, strokeColor=white, strokeWidth=1))     # N-term
    d.add(Rect(70, bar_y, 340, bar_h, fillColor=TEAL, strokeColor=white, strokeWidth=1))       # core
    d.add(Rect(410, bar_y, 30, bar_h, fillColor=ORANGE, strokeColor=white, strokeWidth=1))     # C-term
    for x, t in [(40, "1"), (70, "~20"), (410, "~350"), (440, "376")]:
        S(d, x, bar_y - 11, t, 7, GRAY, "middle")
    S(d, 55, 98, "pLDDT ~30", 7.5, ORANGE); S(d, 240, 98, "pLDDT ~95  (단단한 ATPase core)", 8, TEAL)
    S(d, 425, 98, "~35", 7.5, ORANGE)
    S(d, 55, 26, "흐물 → 관대", 7.5, ORANGE); S(d, 240, 26, "단단 → 결실 치명", 8, TEAL); S(d, 425, 26, "흐물 → 관대", 7.5, ORANGE)
    return d


# ===== Diagram D : junction_orient_cos vectors =====
def diag_orient():
    d = Drawing(CW, 150)
    def panel(x0, title, forward, col, cap):
        d.add(Rect(x0, 16, 220, 124, fillColor=PANEL, strokeColor=LGRAY, strokeWidth=1, rx=8, ry=8))
        S(d, x0 + 12, 122, title, 9.5, TEAL, "start")
        y = 78
        arrow(d, x0 + 25, y, x0 + 75, y, BLUE)               # a: 들어오는 방향
        S(d, x0 + 50, y + 10, "a (p-1로)", 7, INK)
        if forward:
            arrow(d, x0 + 130, y, x0 + 190, y, BLUE)         # b: 같은 방향
        else:
            arrow(d, x0 + 190, y, x0 + 130, y, RED)          # b: 반대
        S(d, x0 + 160, y + 10, "b (q+1에서)", 7, INK)
        S(d, x0 + 110, 34, cap, 8, col, "middle")
    panel(5, "방향 정렬", True, GREEN, "cos ~ +1 → 매끄러운 봉합")
    panel(235, "방향 반대", False, RED, "cos ~ -1 → 꺾임(kink)")
    return d


# ===== Diagram E : correlation bars =====
def diag_corr():
    d = Drawing(CW, 200)
    feats = [("A_plddt", "단단한 core 결실"), ("B_seam_mean_rsasa", "흉터가 표면"),
             ("B_span_external_contacts", "조각이 촘촘히 박힘"), ("B_span_min2_rsasa", "묻힌 잔기 결실"),
             ("B_junction_strain", "재봉합 gap")]
    cx, scale = 315, 175
    d.add(Line(cx, 26, cx, 178, strokeColor=LGRAY, strokeWidth=1))
    S(d, 150, 188, "← 손상 예측", 7.5, RED, "middle"); S(d, 400, 188, "관대 예측 →", 7.5, GREEN, "middle")
    y = 162
    for key, desc in feats:
        rho = CD.get(key, 0.0)
        col = RED if rho < -0.05 else (GREEN if rho > 0.05 else GRAY)
        bx = cx + rho * scale
        d.add(Rect(min(cx, bx), y - 6, abs(rho * scale), 12, fillColor=col, strokeColor=None))
        S(d, 8, y - 3, f"{key}", 7.5, INK, "start")
        S(d, 8, y - 12, desc, 6.8, GRAY, "start")
        tx = bx + (6 if rho >= 0 else -6)
        S(d, tx, y - 3, f"{rho:+.2f}", 7.5, col, ("start" if rho >= 0 else "end"))
        y -= 28
    return d


DIAGS = [("A. 변이 3종", diag_chain_edits), ("B. 재봉합 gap", diag_junction),
         ("C. pLDDT 축", diag_plddt), ("D. orient_cos", diag_orient), ("E. 상관 검증", diag_corr)]


# ---------------------------------------------------------------------------
st = {
    "title": ParagraphStyle("t", fontName=SANS, fontSize=19, textColor=INK, leading=24, spaceAfter=2),
    "sub": ParagraphStyle("s", fontName=SANS, fontSize=10, textColor=GRAY, leading=14, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName=SANS, fontSize=13.5, textColor=TEAL, leading=17, spaceBefore=13, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName=SANS, fontSize=10.5, textColor=INK, leading=14, spaceBefore=6, spaceAfter=2),
    "body": ParagraphStyle("b", fontName=SANS, fontSize=9.5, textColor=INK, leading=15, spaceAfter=5),
    "cap": ParagraphStyle("c", fontName=SANS, fontSize=8.3, textColor=GRAY, leading=11, spaceAfter=8, alignment=1),
    "small": ParagraphStyle("sm", fontName=SANS, fontSize=8.3, textColor=INK, leading=11),
    "smallc": ParagraphStyle("smc", fontName=SANS, fontSize=8.3, textColor=INK, leading=11, alignment=1),
    "hdr": ParagraphStyle("hd", fontName=SANS, fontSize=8.5, textColor=white, leading=11, alignment=1),
}


def P(t, s="body"): return Paragraph(t, st[s])


def tbl(header, rows, widths, aligns=None):
    data = [[Paragraph(h, st["hdr"]) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), st["smallc" if (aligns and aligns[j] == "c") else "small"])
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("GRID", (0, 0), (-1, -1), 0.4, LGRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f5f7f7")]),
    ]))
    return t


def fig(drawing, caption):
    drawing.hAlign = "CENTER"
    return [drawing, P(caption, "cap")]


if "test" in sys.argv:
    story = [P("다이어그램 검증", "title")]
    for name, fn in DIAGS:
        story += [P(name, "h2")] + fig(fn(), "")
    SimpleDocTemplate("_diagtest.pdf", pagesize=(520, 1150), topMargin=20, bottomMargin=20,
                      leftMargin=30, rightMargin=30).build(story)
    print("saved: _diagtest.pdf")
    sys.exit()
reg = R["regression"]; cls = R["classification"]; c = R["counts"]
story = [
    P("Block B — indel 구조 인코더 설명", "title"),
    P("DeepRAD51C / 조원 공유용 / 구조 정보를 벡터로 바꾸는 원리", "sub"),
    HRFlowable(width="100%", thickness=0.6, color=LGRAY, spaceBefore=2, spaceAfter=8),

    P("1. 먼저 — indel은 왜 특별한가", "h1"),
    P("단백질은 아미노산 사슬이 접혀 3D 구조를 이루고, 그 구조가 기능을 한다. 변이가 이 구조를 얼마나 "
      "망가뜨리는지가 곧 기능 손상 정도다. 그런데 <b>indel(삽입/결실)</b>은 <b>SAV(한 아미노산 치환)</b>와 "
      "구조적으로 완전히 다른 사건이다."),
]
story += fig(diag_chain_edits(), "그림 A. SAV는 길이가 그대로지만, 결실은 토막을 잘라 양끝을 다시 잇고, 삽입은 사슬을 늘린다.")
story += [
    P("핵심 차이 4가지", "h2"),
    P("① <b>소수성 코어</b>: 묻힌 잔기는 구조의 기둥, 표면 잔기는 여유가 있다.  "
      "② <b>백본 연결성</b>: 결실은 사슬을 자르고 다시 이어야 하며, 연속된 Cα 간격은 ~3.8A로 고정돼 있다.  "
      "③ <b>2차 구조 주기성</b>: helix/sheet 중간을 건드리면 규칙적 수소결합이 어긋나지만 loop는 유연해 잘 버틴다.  "
      "④ <b>삽입 비대칭</b>: 삽입 잔기는 들어갈 공간이 필요하다."),

    PageBreak(),
    P("2. 재봉합 사건 — Block B의 핵심 아이디어", "h1"),
    P("결실에서 가장 중요한 건 '무엇을 뺐나'보다 <b>'자른 두 끝을 다시 이을 수 있나'</b>이다. "
      "남은 두 잔기(p-1, q+1)가 3D에서 가까우면 쉽게 봉합되고, 멀면 억지로 이어야 해 구조가 깨진다."),
]
story += fig(diag_junction(), "그림 B. 같은 3-residue 결실이라도 표면 loop는 양끝이 가까워 관대, core는 멀어 치명적. "
                              "시작점 하나만 보면 이 차이가 구분되지 않는다.")
story += [
    P("그래서 Block B는 residue를 나열하는 대신 이 <b>재봉합 사건의 구조적 결과</b>를 인코딩한다."),

    P("3. pLDDT를 왜 쓰나 (자주 받는 질문)", "h1"),
    P("pLDDT는 원래 AlphaFold의 <b>예측 신뢰도</b>(0~100)다. 그런데 핵심 반전: "
      "<b>AF가 확신 못 하는 곳은 대개 구조가 원래 흐물거리는 곳</b>이다. 결정구조로 학습한 AF는 구조가 명확한 "
      "곳에서만 확신하고, 무질서/유연 영역에선 확신하지 못한다. 그래서 pLDDT를 사실상 "
      "<b>'단단한가 / 흐물거리는가'(강성) 프록시</b>로 쓴다."),
]
story += fig(diag_plddt(), "그림 C. RAD51C는 양 말단이 흐물거리고(pLDDT ~30) ATPase core가 단단하다(~95). "
                          "단단한 곳(높은 pLDDT) 결실이 훨씬 치명적이다.")
story += [
    P("rSASA와는 다른 축이다", "h2"),
    tbl(["", "측정하는 것", "축"],
        [["rSASA", "묻혔나 / 노출됐나", "공간 (core vs 표면)"],
         ["pLDDT", "단단한가 / 흐물거리나", "동역학 (rigid vs flexible)"]],
        [70, 165, 190], aligns=["c", "l", "l"]),
    P("둘은 관련 있지만(RAD51C에서 상관 -0.66) 동일하진 않다. 표면 loop와 무질서한 꼬리를 rSASA만으론 "
      "구분 못 하는데, pLDDT가 그 유연성 차이를 잡아준다.", "body"),

    PageBreak(),
    P("4. Block B 전체 구성", "h1"),
    P("최종 벡터 = Block A(11, 앵커 한 점 구조) + <b>Block B(11, indel 재봉합)</b> + 타입 one-hot(3) = 25-dim. "
      "Block B는 indel에서만 채워지고 SAV는 전부 0이다.", "body"),
    tbl(["그룹", "feature", "무엇을 / 왜 (생물학)"],
        [["재봉합 기하", "junction_strain", "양끝 Cα 거리 / 3.8A — 직접 봉합 가능성"],
         ["", "junction_orient_cos", "양끝 사슬 방향 정렬 — 매끄럽게 이어지나"],
         ["", "seam_both_loop", "양끝 둘 다 loop인가 (깔끔한 절제)"],
         ["영향 구간", "span_min2_rsasa", "가장 묻힌 2잔기 — 코어를 지우나"],
         ["", "span_max2_plddt", "가장 단단한 2잔기 — 단단한 곳을 지우나"],
         ["", "span_frac_ss", "helix+sheet 비율 — 2차구조 파괴"],
         ["", "span_min_site_dist", "기능 자리(ATP/interface)를 건드리나"],
         ["흉터 자리", "seam_mean_rsasa", "봉합 지점이 묻혔나 / 표면인가"],
         ["연결성", "span_external_contacts", "제거 조각이 나머지와 얼마나 얽혔나"],
         ["크기", "del_len_norm / ins_len_norm", "결실 / 삽입 길이"]],
        [58, 120, 222], aligns=["l", "l", "l"]),
    P("변이 종류는 마지막 one-hot [sav, del, ins]가 지정한다.", "cap"),

    P("5. 계산 디테일 두 가지", "h1"),
    P("<b>junction_orient_cos</b>: 왼쪽 끝으로 들어오는 사슬 방향 a = Cα(p-1)-Cα(p-2), 오른쪽에서 나가는 "
      "방향 b = Cα(q+2)-Cα(q+1)의 코사인. 같은 방향이면 매끄럽게 이어지고 반대면 봉합 시 꺾인다."),
]
story += fig(diag_orient(), "그림 D. orient_cos = 두 사슬 끝 방향벡터의 코사인.")
story += [
    P("<b>min2 (min 대신 최소 2개의 평균)</b>: 구간에서 '가장 묻힌/단단한' 잔기를 볼 때 하나(min)만 보면 "
      "극단치 하나에 흔들린다. 2개 평균으로 robust하게 만들었다. multi-residue indel이 많은 다른 유전자로 "
      "확장할 때 특히 유리하다.", "body"),

    PageBreak(),
    P("6. 실제 데이터로 검증", "h1"),
    P(f"RAD51C SGE 실험값(z_score; 낮을수록 기능 손상)과 각 구조 feature의 상관을 봤다. "
      f"'구조가 파괴적이면 z가 낮아진다'는 예측 방향이 맞으면 성공이다. (결실 {c['del']}개 기준)"),
]
story += fig(diag_corr(), "그림 E. 구조 feature ↔ z_score 상관. 방향이 모두 예측대로다. "
                         "junction_strain만 무신호(아래 한계 참고).")
story += [
    P(f"종합하면 <b>구조 정보만으로</b> z_score를 예측하는 간단한 회귀가 한 번도 안 본 test에서 "
      f"<b>Pearson r = {reg['test']['pearson']:+.2f}</b>, depleted 분류 <b>AUC = {cls['test']['auc']}</b>를 낸다. "
      f"구조 인코딩이 실제 기능 신호를 담고 있다는 뜻이다."),

    P("7. 정직한 한계", "h1"),
    P("① <b>junction_strain / orient_cos 무신호</b>: 이 데이터의 결실이 거의 다 1-residue라 재봉합 gap이 "
      "변이마다 거의 안 변해 변별력이 없다. multi-residue 결실이 많은 유전자에서 살아날 feature라 확장성 위해 유지."),
    P("② <b>span pooling ~ 앵커</b>: 마찬가지로 1-residue 결실에선 구간이 시작점 하나라 span feature가 "
      "Block A 앵커와 같아진다. 이 역시 multi-residue에서만 값을 한다."),
    P("③ pLDDT는 직접 측정이 아니라 <b>모델 프록시</b>다. 더 엄밀히는 결정구조 B-factor나 MD 요동을 쓰지만 "
      "RAD51C엔 없어 실무적 최선으로 AF pLDDT를 쓴다 (실제로 잘 작동)."),
]

if "preview" in sys.argv:
    flat = [Spacer(1, 14) if isinstance(x, PageBreak) else x for x in story]
    SimpleDocTemplate("_preview_all.pdf", pagesize=(595, 2600), leftMargin=18 * mm,
                      rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=14 * mm).build(flat)
    print("saved: _preview_all.pdf")
else:
    SimpleDocTemplate("Block_B_설명.pdf", pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=15 * mm, bottomMargin=14 * mm,
                      title="Block B indel 구조 인코더 설명").build(story)
    print("saved: Block_B_설명.pdf")
