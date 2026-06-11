"""국회 서면질의 답변서 AI 초안 생성 시스템

PDF 형식의 서면질의서를 업로드하면
GPT-4o가 질의 주제를 분류하고 답변 초안을 생성합니다.

실행:
    uv run streamlit run main.py

환경변수 (.env):
    OPENAI_API_KEY
    ASSEMBLY_API_KEY
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 색상 시스템 (Vercel 다크 스타일) ─────────────────────────────────────────
BG          = "#1a1a1a"
BG_CARD     = "#212121"
BG_HOVER    = "#2a2a2a"
BORDER      = "#383838"
BORDER_LIGHT= "#2a2a2a"
TEXT        = "#ededed"
TEXT_MUTED  = "#888888"
TEXT_SUBTLE = "#555555"
ACCENT      = "#888888"
ACCENT_BLUE = "#0070f3"
ACCENT_BLUE_HOVER = "#0060df"
SUCCESS     = "#50e3c2"
WARNING     = "#f5a623"
ERROR       = "#e00"

PAGE_INPUT   = "input"
PAGE_RESULTS = "results"

HIDE_STREAMLIT_STYLE = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="collapsedControl"] {display: none !important;}
section[data-testid="stSidebar"] {display: none !important;}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
    margin-top: 0 !important;
}
.main .block-container { padding-top: 0 !important; margin-top: 0 !important; }
.stMainBlockContainer { padding-top: 0 !important; }
[data-testid="stAppViewContainer"] > section:first-child { padding-top: 0 !important; }
.stApp > div:first-child { padding: 0 !important; margin: 0 !important; }
.stTextArea small, [data-testid="InputInstructions"] { display: none !important; }
</style>
"""

SYSTEM_PROMPT = """당신은 대한민국 기획재정부 기획조정실 기획재정담당관실 소속 행정 전문가입니다.
국회의원이 제출한 서면질문서를 분석하여 아래 JSON 형식으로만 응답하세요.

{
  "title": "질문서 제목 (없으면 내용 기반으로 작성)",
  "assemblyman": "질의 의원 이름",
  "party": "소속 정당",
  "target_minister": "질문 대상 장관",
  "main_topic": "핵심 질의 주제 (한 문장으로 요약)",
  "topics": ["세부 질의 주제1", "세부 질의 주제2", "세부 질의 주제3"],
  "ministry": "주 담당 부처명",
  "related_ministries": ["관련 부처1", "관련 부처2"],
  "urgency": "높음|보통|낮음",
  "urgency_reason": "긴급도 판단 근거",
  "questions": [
    {
      "number": 1,
      "content": "질문 내용 그대로 또는 요약",
      "background": "상황 인식 또는 배경",
      "main_response": "개선 방향 또는 주요 내용 또는 대응 논리",
      "future_plan": "향후 계획 또는 강조 사항 (필요한 경우에만, 불필요하면 빈 문자열)"
    }
  ],
  "overall_draft": "전체 답변서 초안 (아래 형식 준수)",
  "key_points": ["답변 핵심 포인트1", "답변 핵심 포인트2", "답변 핵심 포인트3"],
  "related_laws": ["관련 법령1", "관련 법령2"],
  "additional_info_needed": "추가로 필요한 정보나 검토 사항 (없으면 빈 문자열)"
}

overall_draft 작성 형식:
---
000의원 [소속정당]

[질문 1] 질문 내용 그대로 또는 요약

○ 상황 인식
(현재 상황, 문제점, 관련 통계/현황을 2~3문장으로 작성)

○ 주요 내용 및 대응 논리
(정책 방향, 구체적 대응 방안, 법적 근거를 2~3문장으로 작성)

○ 향후 계획 (필요시)
(향후 추진 계획이나 강조 사항을 1~2문장으로 작성, 불필요하면 생략)


[질문 2] ...
---

규칙:
- overall_draft 첫 줄은 반드시 "000의원 [소속정당]" 형식으로 작성
- 각 질문은 질문 내용 → 상황인식 → 주요내용 → 향후계획 순서
- 향후계획은 필요한 경우에만 작성
- 공문서 형식 (존댓말, 공식 문체)
- 관련 법령 반드시 언급
- urgency 높음: 정치적으로 민감하거나 언론 주목도 높은 경우
- urgency 보통: 일반적인 정책 질의
- urgency 낮음: 단순 현황 파악 질의
- 모르는 내용은 "관련 부처 확인 필요" 명시
- 구체적 수치나 현황 포함하여 작성
"""


# ── 환경변수 ──────────────────────────────────────────────────────────────────

def get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


# ── PDF 텍스트 추출 ───────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as exc:
        raise RuntimeError(f"PDF 텍스트 추출 실패: {exc}") from exc


# ── GPT 분석 ──────────────────────────────────────────────────────────────────

def analyze_question(openai_key: str, text: str) -> dict[str, Any]:
    client = OpenAI(api_key=openai_key)
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"다음 서면질의서를 분석하고 답변 초안을 작성해주세요:\n\n{text[:8000]}"},
        ],
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("GPT-4o 응답이 비어 있습니다.")
    return json.loads(content)


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _html_escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def build_answer_document(result: dict[str, Any], filename: str) -> str:
    lines = [
        "=" * 60,
        "국회 서면질의 답변서 초안",
        "=" * 60,
        "",
        f"■ 질의서 파일: {filename}",
        f"■ 질의 제목: {result.get('title', '-')}",
        f"■ 질의 의원: {result.get('assemblyman', '-')} [{result.get('party', '-')}]",
        f"■ 질문 대상: {result.get('target_minister', '-')}",
        f"■ 핵심 주제: {result.get('main_topic', '-')}",
        f"■ 담당 부처: {result.get('ministry', '-')}",
        f"■ 관련 부처: {', '.join(result.get('related_ministries', []))}",
        f"■ 긴급도: {result.get('urgency', '-')} ({result.get('urgency_reason', '-')})",
        "",
        "─" * 60,
        "【전체 답변서 초안】",
        "─" * 60,
        result.get("overall_draft", ""),
        "",
        "─" * 60,
        "【답변 핵심 포인트】",
        "─" * 60,
    ]
    for i, point in enumerate(result.get("key_points", []), 1):
        lines.append(f"{i}. {point}")

    laws = result.get("related_laws", [])
    if laws:
        lines += ["", "─" * 60, "【관련 법령】", "─" * 60]
        for law in laws:
            lines.append(f"· {law}")

    additional = result.get("additional_info_needed", "")
    if additional:
        lines += ["", "─" * 60, "【추가 검토 필요 사항】", "─" * 60, additional]

    lines += ["", "=" * 60, "※ 본 문서는 AI가 생성한 초안입니다. 반드시 검토 후 사용하세요.", "=" * 60]
    return "\n".join(lines)


# ── 스타일 ────────────────────────────────────────────────────────────────────

def _inject_global_styles() -> None:
    st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
    st.markdown(
        f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', 'Noto Sans KR', sans-serif;
    color: {TEXT};
  }}
  .stApp {{ background-color: {BG} !important; color-scheme: dark; }}

  /* ── 네비게이션 ── */
  .vl-nav {{
    background: rgba(0,0,0,0.8);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid {BORDER};
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
  }}
  .vl-nav-left {{ display: flex; align-items: center; gap: 0.75rem; }}
  .vl-nav-logo {{ font-size: 1rem; font-weight: 700; color: {ACCENT}; letter-spacing: -0.02em; }}
  .vl-nav-divider {{ width: 1px; height: 20px; background: {BORDER}; }}
  .vl-nav-title {{ font-size: 0.875rem; color: {TEXT_MUTED}; }}
  .vl-nav-link {{
    font-size: 0.8125rem; color: {TEXT_MUTED}; text-decoration: none;
    padding: 0.4rem 0.75rem; border: 1px solid {BORDER}; border-radius: 6px;
    transition: all 0.15s;
  }}
  .vl-nav-link:hover {{ color: {ACCENT}; border-color: {TEXT_MUTED}; }}

  /* ── 브레드크럼 ── */
  .vl-breadcrumb {{
    padding: 0.6rem 2rem;
    border-bottom: 1px solid {BORDER_LIGHT};
    font-size: 0.8125rem;
    color: {TEXT_SUBTLE};
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .vl-breadcrumb-left {{ display: flex; align-items: center; gap: 0.4rem; }}
  .vl-breadcrumb strong {{ color: {TEXT_MUTED}; font-weight: 500; }}
  .vl-back-btn {{
    font-size: 0.8125rem; color: {TEXT_MUTED}; text-decoration: none;
    padding: 0.25rem 0.65rem; border: 1px solid {BORDER}; border-radius: 6px;
    transition: all 0.15s;
  }}
  .vl-back-btn:hover {{ color: {ACCENT}; border-color: {TEXT_SUBTLE}; }}

  /* ── 콘텐츠 ── */
  .main .block-container {{
    padding: 3rem 2rem 4rem !important;
    max-width: 800px !important;
    margin: 0 auto !important;
  }}

  /* ── 히어로 ── */
  .vl-hero {{
    text-align: center;
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid {BORDER_LIGHT};
  }}
  .vl-hero-badge {{
    display: inline-block;
    font-size: 0.75rem; font-weight: 500; color: {TEXT_MUTED};
    border: 1px solid {BORDER}; border-radius: 20px;
    padding: 0.2rem 0.75rem; margin-bottom: 1rem;
    letter-spacing: 0.05em; text-transform: uppercase;
  }}
  .vl-hero h2 {{
    font-size: 2rem; font-weight: 700; color: {ACCENT};
    margin: 0 0 0.75rem; letter-spacing: -0.04em; line-height: 1.2;
  }}
  .vl-hero p {{ font-size: 1rem; color: {TEXT_MUTED}; margin: 0; line-height: 1.6; }}

  /* ── 라벨 ── */
  .vl-label {{ display: block; font-size: 0.875rem; font-weight: 500; color: {TEXT_MUTED}; margin: 0 0 0.5rem; }}

  /* ── 파일 업로더 ── */
  [data-testid="stFileUploader"] {{
    background: {BG_CARD} !important;
    border: 1px dashed {BORDER} !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
  }}
  [data-testid="stFileUploader"]:hover {{ border-color: {TEXT_SUBTLE} !important; }}

  /* ── 버튼 ── */
  .stApp .main button,
  .stApp .main [data-testid="stBaseButton-primary"],
  .stApp .main [data-testid="stBaseButton-secondary"],
  .stApp .main div.stButton > button,
  .stApp .main [data-testid="stDownloadButton"] button {{
    background-color: {ACCENT} !important; background: {ACCENT} !important;
    color: {BG} !important; border: 1px solid {ACCENT} !important;
    border-radius: 6px !important; font-weight: 600 !important;
    font-size: 0.875rem !important; box-shadow: none !important; filter: none !important;
  }}
  .stApp .main button:hover,
  .stApp .main [data-testid="stButton"] button:hover,
  .stApp .main [data-testid="stDownloadButton"] button:hover {{
    background-color: #e0e0e0 !important; background: #e0e0e0 !important;
    border-color: #e0e0e0 !important; color: {BG} !important;
  }}
  .stApp .main button:disabled {{
    opacity: 0.4 !important;
    background-color: #333 !important;
    background: #333 !important;
    border-color: #333 !important;
    color: #888 !important;
}}
  .stApp .main div.stButton > button {{
    width: 100% !important; min-height: 44px !important;
    font-size: 0.9375rem !important; font-weight: 600 !important;
  }}

  /* ── 카드 ── */
  .vl-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem;
  }}
  .vl-card-title {{
    font-size: 0.75rem; font-weight: 600; color: {TEXT_MUTED};
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 0 0 1rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid {BORDER_LIGHT};
  }}

  /* ── 정보 행 ── */
  .vl-info-row {{
    display: flex; gap: 1rem; padding: 0.5rem 0;
    border-bottom: 1px solid {BORDER_LIGHT}; font-size: 0.875rem; align-items: flex-start;
  }}
  .vl-info-row:last-child {{ border-bottom: none; }}
  .vl-info-label {{ width: 110px; flex-shrink: 0; color: {TEXT_SUBTLE}; font-size: 0.8125rem; padding-top: 0.1rem; }}
  .vl-info-value {{ color: {TEXT_MUTED}; line-height: 1.5; }}

  /* ── 뱃지 ── */
  .vl-badge {{ display: inline-block; padding: 0.15rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
  .vl-badge-white {{ background: rgba(255,255,255,0.1); color: {TEXT}; border: 1px solid {BORDER}; }}
  .vl-badge-red {{ background: rgba(220,38,38,0.15); color: #f87171; border: 1px solid rgba(220,38,38,0.3); }}
  .vl-badge-yellow {{ background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }}
  .vl-badge-green {{ background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }}

  /* ── 질문 카드 ── */
  .vl-question-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem;
  }}
  .vl-question-header {{
    display: flex; align-items: flex-start; gap: 0.75rem;
    margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid {BORDER_LIGHT};
  }}
  .vl-question-num {{
    width: 24px; height: 24px; background: {ACCENT}; color: {BG};
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; flex-shrink: 0; margin-top: 0.1rem;
  }}
  .vl-question-text {{ font-size: 0.9375rem; color: {TEXT}; line-height: 1.6; font-weight: 500; }}
  .vl-answer-block {{ margin-bottom: 0.85rem; }}
  .vl-answer-block:last-child {{ margin-bottom: 0; }}
  .vl-answer-label {{
    font-size: 0.75rem; font-weight: 600; color: {TEXT_SUBTLE};
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem;
  }}
  .vl-answer-text {{ font-size: 0.875rem; color: {TEXT_MUTED}; line-height: 1.75; }}

  /* ── 답변 초안 ── */
  .vl-draft-box {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 1.25rem 1.5rem; font-size: 0.875rem; line-height: 1.85;
    color: {TEXT_MUTED}; white-space: pre-wrap; font-family: 'Noto Sans KR', sans-serif;
  }}

  /* ── 포인트 ── */
  .vl-point {{
    display: flex; gap: 0.65rem; align-items: flex-start;
    padding: 0.45rem 0; border-bottom: 1px solid {BORDER_LIGHT};
    font-size: 0.875rem; color: {TEXT_MUTED};
  }}
  .vl-point:last-child {{ border-bottom: none; }}
  .vl-point-dot {{ width: 6px; height: 6px; background: {TEXT_SUBTLE}; border-radius: 50%; margin-top: 0.45rem; flex-shrink: 0; }}

  /* ── 경고 ── */
  .vl-warning {{
    background: rgba(245,158,11,0.05); border: 1px solid rgba(245,158,11,0.2);
    border-radius: 8px; padding: 1rem 1.25rem; font-size: 0.875rem;
    color: #fbbf24; line-height: 1.7; margin-bottom: 1rem;
  }}

  /* ── 성공 ── */
  .vl-success {{
    background: rgba(16,185,129,0.05); border: 1px solid rgba(16,185,129,0.2);
    border-radius: 6px; padding: 0.65rem 1rem; font-size: 0.875rem;
    color: #34d399; margin-top: 0.65rem;
  }}

  /* ── 푸터 ── */
  .vl-footer {{
    border-top: 1px solid {BORDER_LIGHT}; padding: 2rem 0;
    text-align: center; font-size: 0.8125rem; color: {TEXT_SUBTLE}; margin-top: 3rem;
  }}
</style>
""",
        unsafe_allow_html=True,
    )


# ── 네비게이션 ────────────────────────────────────────────────────────────────

def _render_nav(*, is_results: bool) -> None:
    back_btn = (
        f'<a class="vl-back-btn" href="javascript:void(0);" '
        f'onclick="window.parent.document.querySelector(\'[data-testid=stBaseButton-secondary]\').click();">← 돌아가기</a>'
        if is_results else
        f'<a class="vl-nav-link" href="https://open.assembly.go.kr" target="_blank" rel="noopener noreferrer">열린국회정보 →</a>'
    )
    st.markdown(
        f"""
<div class="vl-nav">
  <div class="vl-nav-left">
    <span class="vl-nav-logo">▲ AssemblyAI</span>
    <div class="vl-nav-divider"></div>
    <span class="vl-nav-title">국회 서면질의 답변서 시스템</span>
  </div>
  <div>{back_btn}</div>
</div>
<div class="vl-breadcrumb">
  <div class="vl-breadcrumb-left">
    <span>홈</span><span>/</span>
    <span>국회 업무</span><span>/</span>
    <strong>{"분석 결과" if is_results else "서면질의 분석"}</strong>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown(
        f"""
<div class="vl-footer">
  AssemblyAI · 기획재정부 기획조정실 기획재정담당관실<br>
  <span style="margin-top:0.25rem;display:block;">AI가 생성한 초안입니다. 반드시 검토 후 사용하세요.</span>
</div>
""",
        unsafe_allow_html=True,
    )


# ── 세션 ─────────────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults: dict[str, Any] = {
        "page": PAGE_INPUT,
        "analysis_result": None,
        "uploaded_filename": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── 입력 페이지 ───────────────────────────────────────────────────────────────

def _render_input_page(openai_key: str, env_error: bool) -> None:
    st.markdown(
        """
<div class="vl-hero">
  <div class="vl-hero-badge">AI-Powered</div>
  <h2>서면질의서 분석</h2>
  <p>서면질의서 PDF를 업로드하면 질의 주제를 분류하고<br>답변 초안을 자동으로 생성합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if env_error:
        st.error("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")

    st.markdown('<span class="vl-label">서면질의서 PDF 업로드</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "서면질의서 PDF", type=["pdf"], label_visibility="collapsed",
    )

    if uploaded_file:
        st.markdown(
            f'<div class="vl-success">✓ {uploaded_file.name} 업로드 완료</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "분석 시작 →",
        type="primary",
        use_container_width=True,
        disabled=env_error or not uploaded_file,
    ):
        if not uploaded_file:
            st.error("PDF 파일을 업로드하세요.")
            return

        with st.spinner("분석 중..."):
            try:
                text = extract_text_from_pdf(uploaded_file)
                if not text.strip():
                    st.error("PDF에서 텍스트를 추출할 수 없습니다. 스캔 PDF는 지원하지 않습니다.")
                    return

                result = analyze_question(openai_key, text)
                st.session_state.analysis_result = result
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.page = PAGE_RESULTS
                st.rerun()
            except Exception as exc:
                st.error(f"분석 중 오류가 발생했습니다: {exc}")


# ── 결과 페이지 ───────────────────────────────────────────────────────────────

def _render_results_page() -> None:
    result = st.session_state.analysis_result
    filename = st.session_state.uploaded_filename

    if not result:
        st.session_state.page = PAGE_INPUT
        st.rerun()
        return

    urgency = result.get("urgency", "보통")
    urgency_class = {
        "높음": "vl-badge-red",
        "보통": "vl-badge-yellow",
        "낮음": "vl-badge-green",
    }.get(urgency, "vl-badge-yellow")

    # 기본 정보
    st.markdown(
        f"""
<div class="vl-card">
  <div class="vl-card-title">질의서 정보</div>
  <div class="vl-info-row">
    <span class="vl-info-label">파일명</span>
    <span class="vl-info-value">{_html_escape(filename)}</span>
  </div>
  <div class="vl-info-row">
    <span class="vl-info-label">질의 제목</span>
    <span class="vl-info-value">{_html_escape(result.get('title', '-'))}</span>
  </div>
  <div class="vl-info-row">
    <span class="vl-info-label">질의 의원</span>
    <span class="vl-info-value">
      {_html_escape(result.get('assemblyman', '-'))}
      <span class="vl-badge vl-badge-white" style="margin-left:0.4rem;">{_html_escape(result.get('party', '-'))}</span>
    </span>
  </div>
  <div class="vl-info-row">
    <span class="vl-info-label">질문 대상</span>
    <span class="vl-info-value">{_html_escape(result.get('target_minister', '-'))}</span>
  </div>
  <div class="vl-info-row">
    <span class="vl-info-label">핵심 주제</span>
    <span class="vl-info-value">{_html_escape(result.get('main_topic', '-'))}</span>
  </div>
  <div class="vl-info-row">
    <span class="vl-info-label">담당 부처</span>
    <span class="vl-info-value"><span class="vl-badge vl-badge-white">{_html_escape(result.get('ministry', '-'))}</span></span>
  </div>
  <div class="vl-info-row">
    <span class="vl-info-label">관련 부처</span>
    <span class="vl-info-value">{_html_escape(', '.join(result.get('related_ministries', [])) or '-')}</span>
  </div>
  <div class="vl-info-row">
    <span class="vl-info-label">긴급도</span>
    <span class="vl-info-value">
      <span class="vl-badge {urgency_class}">{urgency}</span>
      <span style="font-size:0.8rem;color:{TEXT_SUBTLE};margin-left:0.5rem;">{_html_escape(result.get('urgency_reason', ''))}</span>
    </span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # 질문별 답변
    questions = result.get("questions", [])
    if questions:
        st.markdown(
            f'<div style="font-size:0.75rem;font-weight:600;color:{TEXT_SUBTLE};text-transform:uppercase;letter-spacing:0.08em;margin:1.25rem 0 0.75rem;">질문별 답변 초안 ({len(questions)}건)</div>',
            unsafe_allow_html=True,
        )
        for q in questions:
            future_html = ""
            if q.get("future_plan"):
                future_html = f"""
<div class="vl-answer-block">
  <div class="vl-answer-label">향후 계획</div>
  <div class="vl-answer-text">{_html_escape(q.get('future_plan', ''))}</div>
</div>
"""
            st.markdown(
                f"""
<div class="vl-question-card">
  <div class="vl-question-header">
    <span class="vl-question-num">{q.get('number', '-')}</span>
    <span class="vl-question-text">{_html_escape(q.get('content', '-'))}</span>
  </div>
  <div class="vl-answer-block">
    <div class="vl-answer-label">상황 인식</div>
    <div class="vl-answer-text">{_html_escape(q.get('background', '-'))}</div>
  </div>
  <div class="vl-answer-block">
    <div class="vl-answer-label">주요 내용 및 대응 논리</div>
    <div class="vl-answer-text">{_html_escape(q.get('main_response', '-'))}</div>
  </div>
  {future_html}
</div>
""",
                unsafe_allow_html=True,
            )

    # 전체 답변서 초안
    draft = result.get("overall_draft", "")
    st.markdown(
        f"""
<div class="vl-card" style="margin-top:1rem;">
  <div class="vl-card-title">전체 답변서 초안</div>
  <div class="vl-draft-box">{_html_escape(draft)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # 핵심 포인트 + 관련 법령
    col1, col2 = st.columns(2)
    with col1:
        points_html = "".join(
            f'<div class="vl-point"><span class="vl-point-dot"></span><span>{_html_escape(p)}</span></div>'
            for p in result.get("key_points", [])
        )
        st.markdown(
            f"""
<div class="vl-card">
  <div class="vl-card-title">핵심 포인트</div>
  {points_html or f'<span style="color:{TEXT_SUBTLE};font-size:0.875rem;">없음</span>'}
</div>
""",
            unsafe_allow_html=True,
        )
    with col2:
        laws_html = "".join(
            f'<div class="vl-point"><span class="vl-point-dot"></span><span>{_html_escape(l)}</span></div>'
            for l in result.get("related_laws", [])
        )
        st.markdown(
            f"""
<div class="vl-card">
  <div class="vl-card-title">관련 법령</div>
  {laws_html or f'<span style="color:{TEXT_SUBTLE};font-size:0.875rem;">없음</span>'}
</div>
""",
            unsafe_allow_html=True,
        )

    # 추가 검토 사항
    additional = result.get("additional_info_needed", "")
    if additional:
        st.markdown(
            f"""
<div class="vl-warning">
  ⚠ 추가 검토 필요 사항<br>
  <span style="font-weight:400;margin-top:0.35rem;display:block;">{_html_escape(additional)}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    # 다운로드
    st.markdown("<br>", unsafe_allow_html=True)
    doc_text = build_answer_document(result, filename)
    st.download_button(
        "답변서 초안 저장 (.txt)",
        data=doc_text,
        file_name=f"답변서초안_{filename.replace('.pdf', '').replace('.PDF', '')}.txt",
        mime="text/plain",
        type="secondary",
        use_container_width=True,
    )

    # 이전으로 버튼 (숨김)
    if st.button("← 이전으로 돌아가기", type="secondary", key="btn_go_back"):
        st.session_state.page = PAGE_INPUT
        st.rerun()

    _render_footer()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="AssemblyAI",
        page_icon="▲",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_session()
    _inject_global_styles()

    openai_key = get_openai_key()
    env_error = not openai_key

    is_results = st.session_state.page == PAGE_RESULTS
    _render_nav(is_results=is_results)

    if is_results:
        _render_results_page()
    else:
        _render_input_page(openai_key, env_error)


if __name__ == "__main__":
    main()