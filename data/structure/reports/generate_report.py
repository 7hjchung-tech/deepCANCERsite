# -*- coding: utf-8 -*-
"""generate_report.py — 진행상황 PDF 리포트 생성 (reportlab, 한글 CID 폰트)."""

import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, HRFlowable)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
SANS, SERIF = "HYGothic-Medium", "HYSMyeongJo-Medium"

R = json.load(open("analysis_results.json"))

INK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#0f6d6d")
LIGHT = colors.HexColor("#e8f2f2")
GRAY = colors.HexColor("#666666")
ROW = colors.HexColor("#f5f7f7")

st = {
    "title": ParagraphStyle("t", fontName=SANS, fontSize=20, textColor=INK, leading=25, spaceAfter=2),
    "sub":   ParagraphStyle("s", fontName=SANS, fontSize=10, textColor=GRAY, leading=14, spaceAfter=10),
    "h1":    ParagraphStyle("h1", fontName=SANS, fontSize=14, textColor=ACCENT, leading=18,
                            spaceBefore=14, spaceAfter=6),
    "h2":    ParagraphStyle("h2", fontName=SANS, fontSize=11, textColor=INK, leading=15,
                            spaceBefore=8, spaceAfter=3),
    "body":  ParagraphStyle("b", fontName=SANS, fontSize=9.5, textColor=INK, leading=15, spaceAfter=5),
    "small": ParagraphStyle("sm", fontName=SANS, fontSize=8.3, textColor=INK, leading=11),
    "smallc":ParagraphStyle("smc", fontName=SANS, fontSize=8.3, textColor=INK, leading=11, alignment=1),
    "hdr":   ParagraphStyle("hd", fontName=SANS, fontSize=8.5, textColor=colors.white, leading=11, alignment=1),
    "code":  ParagraphStyle("cd", fontName=SANS, fontSize=8.2, textColor=colors.HexColor("#0a3d3d"),
                            leading=13, backColor=LIGHT, leftIndent=6, rightIndent=6,
                            spaceBefore=3, spaceAfter=6, borderPadding=5),
}


def P(t, s="body"):
    return Paragraph(t, st[s])


def tbl(header, rows, widths, aligns=None):
    data = [[Paragraph(h, st["hdr"]) for h in header]]
    for r in rows:
        cells = []
        for j, c in enumerate(r):
            sty = "smallc" if (aligns and aligns[j] == "c") else "small"
            cells.append(Paragraph(str(c), st[sty]))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd8d8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
    ]
    t.setStyle(TableStyle(style))
    return t


def hr():
    return HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cccccc"),
                      spaceBefore=4, spaceAfter=8)


F = []  # flowables

# ===== 표지 =====
F += [P("DeepRAD51C — 구조 정보 Encoding 진행상황", "title"),
      P("RAD51C 변이 효과 예측 모델의 구조 feature 파트 / 작성일 2026-07-12", "sub"),
      hr()]

# ===== 1. 프로젝트 한눈에 =====
F += [P("1. 프로젝트 한눈에", "h1"),
      P("RAD51C(376 aa) 단백질의 변이가 기능에 미치는 영향을 예측하는 모델을 만드는 중이며, "
        "그중 <b>구조 정보를 벡터로 인코딩</b>하는 부분을 이번에 구현했다. "
        "다루는 변이는 <b>SAV(단일 아미노산 치환, missense)</b>와 <b>in-frame indel(결실/삽입)</b> 두 종류다. "
        "nonsense / frameshift / splice / intron / synonymous는 범위에서 제외했고, "
        "이번 인코딩은 <b>구조 feature만</b> 사용한다(ESM 임베딩 / LLR/PLLR은 넣지 않음)."),
      P("각 변이는 최종적으로 <b>25차원 구조 feature 벡터</b>로 표현된다:", "body")]

F += [tbl(["블록", "차원", "내용", "언제 채워지나"],
          [["Block A", "11", "앵커 residue 한 점의 구조 (pLDDT, 2차구조, 표면노출, 기능자리 6곳까지 거리)",
            "모든 변이 (indel=시작위치, SAV=치환위치)"],
           ["Block B", "11", "indel '자르고 다시 잇는 사건'의 구조적 결과 (재봉합 기하 + 영향 구간 + 흉터 + 연결성)",
            "indel만 (SAV는 전부 0)"],
           ["one-hot", "3", "[sav, del, ins] 3-way 구분", "모든 변이"]],
          [58, 30, 250, 130], aligns=["c", "c", "l", "l"])]

F += [P("Block B(11-dim) 세부 구성", "h2"),
      tbl(["그룹", "dim", "주요 feature"],
          [["① 재봉합 기하", "3", "junction_strain, junction_orient_cos, seam_both_loop"],
           ["② 영향 구간 (span pooling)", "4", "span_min2_rsasa, span_max2_plddt, span_frac_ss, span_min_site_dist"],
           ["③ 흉터 자리", "1", "seam_mean_rsasa"],
           ["④ 연결성", "1", "span_external_contacts"],
           ["⑤ 크기", "2", "del_len_norm, ins_len_norm"]],
          [95, 28, 345], aligns=["l", "c", "l"])]

# ===== 2. 핵심 결과 =====
c = R["counts"]; sc = R["split_counts"]; reg = R["regression"]; cls = R["classification"]
F += [P("2. 핵심 결과", "h1"),
      P("2.1 데이터", "h2"),
      P(f"scope 적용 후 총 <b>{R['n_total']}개</b> 변이: "
        f"SAV {c['sav']} / 결실 {c['del']} / 삽입 {c['ins']}. "
        f"(split: train {sc['train']} / val {sc['val']} / test {sc['test']}, position 기반 분할 유지)")]

# 2.2 검증 상관
cd = {n: v for n, v in R["corr_del"]}
top = ["A_plddt", "B_seam_mean_rsasa", "B_span_external_contacts", "B_span_min2_rsasa", "B_junction_strain"]
F += [P("2.2 검증 — 결실에서 구조 feature가 실제 z_score와 상관있나", "h2"),
      P("z_score가 낮을수록 기능 손상(depleted). '구조가 더 파괴적이면 z가 낮아진다'는 예측이 맞는지 확인.", "body"),
      tbl(["feature", "ρ (결실)", "해석"],
          [[n, f"{cd[n]:+.3f}",
            {"A_plddt": "잘 접힌 core 결실 → 손상",
             "B_seam_mean_rsasa": "표면 흉터 → 관대",
             "B_span_external_contacts": "촘촘히 박힌 조각 결실 → 손상 (Block B 핵심)",
             "B_span_min2_rsasa": "묻힌 residue 결실 → 손상",
             "B_junction_strain": "무신호 — 결실이 대부분 1-residue"}[n]] for n in top],
          [175, 70, 215], aligns=["l", "c", "l"])]

# 2.3 baseline
F += [P("2.3 구조-only 예측 baseline (Ridge 회귀, torch 없이)", "h2"),
      P("구조 25-dim만으로 z_score를 예측. train으로 학습, 한 번도 안 본 test로 평가.", "body"),
      tbl(["구분", "n", "Pearson r", "Spearman ρ"],
          [["train", reg["train"]["n"], f"{reg['train']['pearson']:+.3f}", f"{reg['train']['spearman']:+.3f}"],
           ["val", reg["val"]["n"], f"{reg['val']['pearson']:+.3f}", f"{reg['val']['spearman']:+.3f}"],
           ["<b>test</b>", reg["test"]["n"], f"<b>{reg['test']['pearson']:+.3f}</b>", f"{reg['test']['spearman']:+.3f}"],
           ["test / SAV", reg["test_by_type"]["sav"]["n"], f"{reg['test_by_type']['sav']['pearson']:+.3f}", f"{reg['test_by_type']['sav']['spearman']:+.3f}"],
           ["test / 결실", reg["test_by_type"]["del"]["n"], f"{reg['test_by_type']['del']['pearson']:+.3f}", f"{reg['test_by_type']['del']['spearman']:+.3f}"]],
          [110, 60, 100, 100], aligns=["l", "c", "c", "c"]),
      Spacer(1, 4),
      P(f"분류(depleted vs unchanged): 같은 구조 예측 점수로 test <b>AUC = {cls['test']['auc']}</b> "
        f"(n={cls['test']['n']}). → 구조 정보만으로도 기능 손상 변이를 꽤 잘 가려낸다.", "body")]

F += [PageBreak()]

# ===== 3. 파일 안내 =====
F += [P("3. 파일 안내 — 각 파일이 무엇인가", "h1"),
      P("코드 파일 (code/)", "h2"),
      tbl(["파일", "역할"],
          [["block_b.py", "핵심. build_block_a()로 구조 11-dim 테이블 계산, BlockBEncoder로 indel 11-dim 인코딩, encode_full()로 25-dim 조립."],
           ["assemble.py", "조립기 데모. sav/del/ins 예시가 올바른 25-dim이 되는지 확인."],
           ["build_dataset.py", "split_manifest.csv를 파싱해 전체 [4914,25] feature 표 생성."],
           ["analysis.py", "구조 feature 진단 + Ridge 예측 baseline (분석 결과 JSON 저장)."],
           ["generate_report.py", "이 PDF 리포트 생성."],
           ["explain_blockb.py", "조원 설명용 그림 PDF(Block_B_설명.pdf) 생성."]],
          [130, 340], aligns=["l", "l"]),
      P("결과 파일 (results/)", "h2"),
      tbl(["파일", "내용"],
          [["rad51c_X.npy", "모델 입력. [4914, 25] float 구조 feature 행렬."],
           ["rad51c_y.npy", "라벨. [4914] z_score_D4_D14 (회귀 타깃)."],
           ["rad51c_struct_features.csv", "사람용 통합표. var_id / split / 라벨 + 25 feature."],
           ["rad51c_meta.csv", "변이 메타 (var_id, split, type, 위치, 길이, 라벨)."],
           ["rad51c_blockA.csv", "residue별 Block A 12-dim 구조 테이블 (376행)."],
           ["analysis_results.json", "분석 수치 (상관 / baseline / AUC) 원본."]],
          [150, 320], aligns=["l", "l"]),
      P("입력 파일 (inputs/)", "h2"),
      tbl(["파일", "내용"],
          [["AF-O43502-F1.pdb", "RAD51C AlphaFold WT 구조 (모든 구조 feature의 원천)."],
           ["rad51c_residue_annotation.csv", "기능 자리(Walker A/B, ATP, ssDNA, BCDX2/CX3) 정의 — 거리 계산용."],
           ["split_manifest.csv", "SGE 실험 데이터: 변이 위치 / 타입 / 변이서열 / z_score / train/val/test 분할."],
           ["wt_sequence.txt", "RAD51C WT 아미노산 서열 (376 aa, 검증용)."]],
          [150, 320], aligns=["l", "l"])]

# ===== 4. 설계 결정 =====
od = R["orient_diag"]
F += [P("4. 설계 결정과 근거", "h1"),
      P("• <b>구조만 사용</b> — 이 파트는 ESM 임베딩 / LLR과 독립적으로 구조 신호만 담당. 디버깅 / ablation이 깔끔.", "body"),
      P("• <b>Block B = '재봉합 사건' 인코더(Direction 2)</b> — indel은 SAV와 달리 사슬을 자르고 다시 잇는 사건. "
        "residue를 나열(pooling)하지 않고, 그 사건의 구조적 대가(끊긴 두 끝의 거리, 잘려나간 조각의 박힘 정도 등)를 인코딩. "
        "가장 indel다운 정보를 컴팩트하게 담아 SAV=0 스킴과도 궁합이 좋음.", "body"),
      P("• <b>SAV는 Block B를 0으로</b> — 치환은 점 하나(Block A)로 변이 전체가 설명되므로 추가 구조 정보가 없음. "
        "SAV/indel 구분은 바깥 one-hot이 담당.", "body"),
      P("• <b>삽입 feature 유지</b> — 이 데이터엔 삽입이 2개뿐이라 거의 0이지만, 향후 다른 유전자(삽입 많은 경우)로 "
        "확장할 때를 위해 일반 indel 인코더로 남겨둠.", "body"),
      P(f"• <b>정직한 한계</b> — junction_strain / junction_orient_cos는 이 데이터에서 무신호 "
        f"(orient_cos 결실 ρ={od['del_spearman_z']}). 결실이 거의 다 1-residue라 '재봉합 gap'이 변이마다 "
        f"거의 안 변하기 때문. multi-residue 결실이 많은 유전자에선 살아날 feature라 유지.", "body")]

# ===== 5. 앞으로 할 일 =====
F += [P("5. 앞으로 하면 좋을 것 (우선순위 순)", "h1"),
      tbl(["할 일", "설명"],
          [["1. 다운스트림 MLP 연결", "이 [N,25]을 문서 아키텍처의 구조 MLP 블록에 넣어 실제 예측까지. torch 필요(무거워짐, GPU 서버 권장)."],
           ["2. 다른 feature 블록과 통합", "ESM WT-difference 임베딩 등 다른 입력 블록과 concat하는 전체 파이프라인 조립."],
           ["3. junction_orient_cos 정리", "이 데이터에서 무신호 — 유지하되, 최종 모델에서 ablation으로 기여도 재확인."],
           ["4. 외부 검증", "Hu et al.(2023), Prakash et al.(2024)의 HR proficiency assay로 일반화 확인."],
           ["5. 다른 유전자 확장", "BRCA-paralog, FANC 등 짧은 유전자로 framework 확장. 삽입 / multi-residue 결실 많은 데이터에서 Block B 진가 재확인."]],
          [130, 340], aligns=["l", "l"])]

# ===== 6. 재현 방법 =====
F += [P("6. 재현 방법", "h1"),
      P("의존성: numpy, pandas, biotite (구조 인코딩) / reportlab (PDF). torch / scipy 불필요.", "body"),
      P("inputs/의 파일들을 작업 폴더에 두고 순서대로 실행:", "body"),
      P("python build_dataset.py&nbsp;&nbsp;# split_manifest.csv → rad51c_X.npy, rad51c_y.npy, features.csv<br/>"
        "python analysis.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# → analysis_results.json (상관 / baseline / AUC)<br/>"
        "python generate_report.py&nbsp;# → 이 PDF", "code")]

# ===== 부록: 전체 상관표 =====
F += [PageBreak(), P("부록. 25개 feature 전체 — z_score Spearman 상관", "h1"),
      P("전체 / SAV / 결실 각각에서의 상관. (SAV는 Block B가 0이라 B_* 상관은 정의되지 않음/0)", "body")]
ca = {n: v for n, v in R["corr_all"]}
cs = {n: v for n, v in R["corr_sav"]}
rows = []
for n, _ in R["corr_all"]:
    rows.append([n, f"{ca[n]:+.3f}", f"{cs[n]:+.3f}", f"{cd[n]:+.3f}"])
F += [tbl(["feature", "전체 ρ", "SAV ρ", "결실 ρ"], rows,
          [230, 75, 75, 75], aligns=["l", "c", "c", "c"])]

doc = SimpleDocTemplate("진행상황_리포트.pdf", pagesize=A4,
                        leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=16 * mm, bottomMargin=16 * mm,
                        title="DeepRAD51C 구조 encoding 진행상황")
doc.build(F)
print("saved: 진행상황_리포트.pdf")
