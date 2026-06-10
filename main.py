"""국회 서면질의 답변서 AI 초안 생성 시스템

PDF 형식의 서면질의서를 업로드하면
GPT-4o가 질의 주제를 분류하고 답변 초안을 생성합니다.

실행:
    uv run streamlit run main.py

환경변수 (.env):
    OPENAI_API_KEY, 
    ASSEMBLY_API_KEY
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ── 색상 시스템 ───────────────────────────────────────────────────────────────
PRIMARY       = "#0062B1"
PRIMARY_DARK  = "#004A8C"
PRIMARY_LIGHT = "#E8F0FA"
WHITE         = "#FFFFFF"
GRAY_50       = "#F7F8FA"
GRAY_100      = "#EEF0F4"
GRAY_200      = "#DDE1E9"
GRAY_400      = "#9AA4B2"
GRAY_600      = "#5A6475"
GRAY_900      = "#1A2130"

PAGE_INPUT   = "input"
PAGE_RESULTS = "results"

# ── 담당 부처 목록 ────────────────────────────────────────────────────────────
MINISTRY_OPTIONS = [
    "기획재정부", "교육부", "과학기술정보통신부", "외교부", "통일부",
    "법무부", "국방부", "행정안전부", "문화체육관광부", "농림축산식품부",
    "산업통상자원부", "보건복지부", "환경부", "고용노동부", "여성가족부",
    "국토교통부", "해양수산부", "중소벤처기업부", "국무조정실", "기타",
]

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


[질문 2] 질문 내용 그대로 또는 요약

○ 상황 인식
...

○ 주요 내용 및 대응 논리
...

○ 향후 계획 (필요시)
...
---

규칙:
- overall_draft 첫 줄은 반드시 "000의원 [소속정당]" 형식으로 작성
- 그 다음 줄부터 질문별로 구분하여 작성
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


def validate_openai_key(key: str) -> str | None:
    """OpenAI API 키 형식 검증. 문제가 있으면 오류 메시지 반환."""
    if not key:
        return "OPENAI_API_KEY가 .env에 설정되어 있지 않습니다."
    if key == os.getenv("ASSEMBLY_API_KEY", "").strip():
        return (
            "OPENAI_API_KEY에 AssemblyAI 키가 들어가 있습니다. "
            ".env에서 OPENAI_API_KEY는 OpenAI 키(sk- 또는 sk-proj-로 시작)여야 합니다."
        )
    if not key.startswith(("sk-", "sk-proj-")):
        return (
            "OPENAI_API_KEY 형식이 올바르지 않습니다. "
            "OpenAI 대시보드(https://platform.openai.com/api-keys)에서 발급한 키를 사용하세요."
        )
    return None


# ── PDF 텍스트 추출 ───────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    """PDF에서 텍스트 추출."""
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
    """GPT-4o로 질의서 분석 및 답변 초안 생성."""
    import json

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


def _format_question_answer(q: dict[str, Any]) -> str:
    """질문별 답변 초안 텍스트 생성."""
    parts: list[str] = []
    for label, key in (
        ("○ 상황 인식", "background"),
        ("○ 주요 내용 및 대응 논리", "main_response"),
        ("○ 향후 계획", "future_plan"),
    ):
        value = str(q.get(key, "")).strip()
        if value:
            parts.append(f"{label}\n{value}")
    if parts:
        return "\n\n".join(parts)
    return str(q.get("draft_answer", "")).strip()


def _render_question_answer_html(q: dict[str, Any]) -> str:
    """질문별 답변 초안 HTML 생성."""
    sections: list[str] = []
    for label, key in (
        ("○ 상황 인식", "background"),
        ("○ 주요 내용 및 대응 논리", "main_response"),
        ("○ 향후 계획", "future_plan"),
    ):
        value = str(q.get(key, "")).strip()
        if value:
            sections.append(
                f'<p class="krds-question-answer-label">{label}</p>'
                f'<div class="krds-draft-box">{_html_escape(value)}</div>'
            )
    if sections:
        return "".join(sections)
    draft = str(q.get("draft_answer", "")).strip()
    if draft:
        return (
            '<p class="krds-question-answer-label">개별 답변 초안</p>'
            f'<div class="krds-draft-box">{_html_escape(draft)}</div>'
        )
    return f'<p style="color:{GRAY_400};">답변 초안을 생성할 수 없습니다.</p>'


def build_answer_document(result: dict[str, Any], filename: str) -> str:
    """답변서 텍스트 문서 생성."""
    urgency = result.get("urgency", "-")
    urgency_reason = result.get("urgency_reason", "")
    lines = [
        "=" * 60,
        "국회 서면질의 답변서 초안",
        "=" * 60,
        "",
        f"■ 질의서 파일: {filename}",
        f"■ 질의 제목: {result.get('title', '-')}",
        f"■ 질의 의원: {result.get('assemblyman', '-')}",
        f"■ 소속 정당: {result.get('party', '-')}",
        f"■ 질문 대상 장관: {result.get('target_minister', '-')}",
        f"■ 담당 부처: {result.get('ministry', '-')}",
        f"■ 긴급도: {urgency}" + (f" ({urgency_reason})" if urgency_reason else ""),
        "",
        "─" * 60,
        "【세부 질의 주제】",
        "─" * 60,
    ]
    for i, topic in enumerate(result.get("topics", []), 1):
        lines.append(f"{i}. {topic}")

    questions = result.get("questions", [])
    if questions:
        lines += ["", "─" * 60, "【질문별 답변 초안】", "─" * 60]
        for q in questions:
            num = q.get("number", "")
            lines += [
                "",
                f"▶ 질문 {num}",
                q.get("content", ""),
                "",
                "[답변 초안]",
                _format_question_answer(q),
            ]

    lines += [
        "",
        "─" * 60,
        "【전체 답변서 초안】",
        "─" * 60,
        result.get("overall_draft", result.get("draft_answer", "")),
        "",
        "─" * 60,
        "【답변 핵심 포인트】",
        "─" * 60,
    ]
    for i, point in enumerate(result.get("key_points", []), 1):
        lines.append(f"{i}. {point}")

    related_laws = result.get("related_laws", [])
    if related_laws:
        lines += ["", "─" * 60, "【관련 법령】", "─" * 60]
        for i, law in enumerate(related_laws, 1):
            lines.append(f"{i}. {law}")

    additional = result.get("additional_info_needed", "")
    if additional:
        lines += [
            "",
            "─" * 60,
            "【추가 검토 필요 사항】",
            "─" * 60,
            additional,
        ]

    lines += ["", "=" * 60, "※ 본 문서는 AI가 생성한 초안입니다. 반드시 검토 후 사용하세요.", "=" * 60]
    return "\n".join(lines)


# ── 스타일 ────────────────────────────────────────────────────────────────────

def _inject_global_styles() -> None:
    st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
    st.markdown(
        f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] {{
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    color: {GRAY_900};
  }}
  .stApp {{ background-color: {GRAY_50} !important; color-scheme: light; }}

  /* ── GNB ── */
  .krds-gnb {{
    background: {PRIMARY}; height: 40px; display: flex; align-items: center;
    justify-content: flex-end; padding: 0 2rem; margin: 0;
  }}
  .krds-gnb a {{ color: rgba(255,255,255,0.9); font-size: 0.8rem; text-decoration: none; }}
  .krds-gnb a:hover {{ color: #fff; text-decoration: underline; }}

  /* ── 헤더 ── */
  .krds-header {{
    background: {WHITE}; border-bottom: 3px solid {PRIMARY};
    padding: 1.25rem 2rem; display: flex; align-items: center;
    justify-content: space-between; margin: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  .krds-header-left {{ display: flex; align-items: center; gap: 1rem; }}
  .krds-title-wrap h1 {{
    margin: 0; font-size: 1.375rem; font-weight: 700;
    color: {PRIMARY}; letter-spacing: -0.02em; line-height: 1.3;
  }}
  .krds-title-wrap p {{ margin: 0.2rem 0 0; font-size: 0.8125rem; color: {GRAY_600}; }}

  /* ── 브레드크럼 ── */
  .krds-breadcrumb {{
    background: {GRAY_100}; border-bottom: 1px solid {GRAY_200};
    padding: 0.6rem 2rem; font-size: 0.8125rem; color: {GRAY_600};
    display: flex; align-items: center; justify-content: space-between; margin: 0;
  }}
  .krds-breadcrumb-left {{ display: flex; align-items: center; gap: 0.4rem; }}
  .krds-breadcrumb strong {{ color: {PRIMARY}; font-weight: 600; }}
  .krds-back-btn {{
    font-size: 0.8125rem; color: {PRIMARY}; font-weight: 600;
    text-decoration: none; display: flex; align-items: center; gap: 0.3rem;
    padding: 0.2rem 0.65rem; border: 1px solid {PRIMARY};
    border-radius: 4px; background: {WHITE};
  }}
  .krds-back-btn:hover {{ background: {PRIMARY_LIGHT}; }}

  /* ── 콘텐츠 ── */
  .main .block-container {{
    padding: 2rem 4rem 3rem !important;
    max-width: 960px !important;
    margin: 0 auto !important;
  }}

  /* ── 히어로 ── */
  .krds-input-hero {{
    text-align: center; margin-bottom: 2rem;
    padding-bottom: 1.5rem; border-bottom: 1px solid {GRAY_200};
  }}
  .krds-input-hero h2 {{
    font-size: 1.5rem; font-weight: 700; color: {GRAY_900};
    margin: 0 0 0.5rem; letter-spacing: -0.02em;
  }}
  .krds-input-hero p {{ font-size: 0.9375rem; color: {GRAY_600}; margin: 0; }}

  /* ── 라벨 ── */
  .krds-label {{
    display: block; font-size: 0.875rem; font-weight: 600;
    color: {GRAY_900}; margin: 0 0 0.4rem;
  }}

  /* ── 버튼 ── */
  .stApp .main button,
  .stApp .main [data-testid="stFormSubmitButton"] button,
  .stApp .main [data-testid="stBaseButton-primary"],
  .stApp .main [data-testid="stBaseButton-secondary"],
  .stApp .main div.stButton > button,
  .stApp .main .stFormSubmitButton button,
  .stApp .main [data-testid="stDownloadButton"] button {{
    background-color: {PRIMARY} !important; background: {PRIMARY} !important;
    color: {WHITE} !important; border: 1px solid {PRIMARY} !important;
    border-radius: 6px !important; font-weight: 600 !important;
    box-shadow: none !important; filter: none !important;
  }}
  .stApp .main button:hover,
  .stApp .main [data-testid="stButton"] button:hover,
  .stApp .main [data-testid="stDownloadButton"] button:hover {{
    background-color: {PRIMARY_DARK} !important; background: {PRIMARY_DARK} !important;
    border-color: {PRIMARY_DARK} !important; color: {WHITE} !important;
  }}
  .stApp .main button:disabled {{ opacity: 0.5 !important; }}
  .stApp .main [data-testid="stFormSubmitButton"] button {{
    width: 100% !important; min-height: 48px !important;
    font-size: 1rem !important; font-weight: 700 !important;
  }}

  /* ── 결과 카드 ── */
  .krds-result-section {{
    background: {WHITE}; border: 1px solid {GRAY_200};
    border-radius: 8px; padding: 1.5rem;
    margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }}
  .krds-section-heading {{
    font-size: 1rem; font-weight: 700; color: {GRAY_900};
    margin: 0 0 0.85rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid {PRIMARY};
    display: flex; align-items: center; gap: 0.5rem;
  }}

  /* ── 정보 뱃지 ── */
  .krds-badge {{
    display: inline-block; padding: 0.2rem 0.6rem;
    border-radius: 4px; font-size: 0.8125rem; font-weight: 600;
  }}
  .krds-badge-blue {{ background: {PRIMARY_LIGHT}; color: {PRIMARY}; }}
  .krds-badge-red {{ background: #FFF5F5; color: #C53030; }}
  .krds-badge-yellow {{ background: #FFFAF0; color: #C05621; }}
  .krds-badge-green {{ background: #F0FFF4; color: #276749; }}

  /* ── 답변 초안 박스 ── */
  .krds-draft-box {{
    background: {GRAY_50}; border: 1px solid {GRAY_200};
    border-radius: 6px; padding: 1.25rem 1.5rem;
    font-size: 0.9375rem; line-height: 1.8; color: {GRAY_900};
    white-space: pre-wrap;
  }}

  /* ── 질문 카드 ── */
  .krds-question-card {{
    background: {WHITE}; border: 1px solid {GRAY_200};
    border-radius: 8px; padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
  }}
  .krds-question-card:last-child {{ margin-bottom: 0; }}
  .krds-question-num {{
    display: inline-block; background: {PRIMARY}; color: {WHITE};
    font-size: 0.8125rem; font-weight: 700;
    padding: 0.2rem 0.65rem; border-radius: 4px; margin-bottom: 0.75rem;
  }}
  .krds-question-content {{
    font-size: 0.9375rem; font-weight: 600; color: {GRAY_900};
    margin-bottom: 0.75rem; line-height: 1.6;
  }}
  .krds-question-answer-label {{
    font-size: 0.8125rem; font-weight: 600; color: {GRAY_600};
    margin-bottom: 0.4rem;
  }}

  /* ── 포인트 리스트 ── */
  .krds-point {{
    display: flex; gap: 0.75rem; align-items: flex-start;
    padding: 0.6rem 0; border-bottom: 1px solid {GRAY_100};
    font-size: 0.875rem; color: {GRAY_600};
  }}
  .krds-point:last-child {{ border-bottom: none; }}
  .krds-point-num {{
    background: {PRIMARY}; color: {WHITE};
    width: 22px; height: 22px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; flex-shrink: 0;
  }}

  /* ── 푸터 ── */
  .krds-footer {{
    background: {GRAY_900}; color: rgba(255,255,255,0.7); text-align: center;
    padding: 1.25rem 1rem; font-size: 0.8125rem; margin-top: 3rem; line-height: 1.6;
  }}
</style>
""",
        unsafe_allow_html=True,
    )


# ── 헤더 ─────────────────────────────────────────────────────────────────────

def _render_site_chrome(*, is_results: bool) -> None:
    back_btn = (
        f'<a class="krds-back-btn" href="javascript:void(0);" '
        f'onclick="window.parent.document.querySelector(\'[data-testid=stBaseButton-secondary]\').click();">'
        f'← 이전으로 돌아가기</a>'
        if is_results else ""
    )
    st.markdown(
        f"""
<div class="krds-gnb">
  <a href="https://open.assembly.go.kr" target="_blank" rel="noopener noreferrer">열린국회정보 바로가기 →</a>
</div>
<div class="krds-header">
  <div class="krds-header-left">
    <div class="krds-title-wrap">
      <h1>국회 서면질의 답변서 AI 초안 생성 시스템</h1>
      <p>GPT-4o 기반 질의 분류 및 답변 초안 자동 생성</p>
    </div>
  </div>
</div>
<div class="krds-breadcrumb">
  <div class="krds-breadcrumb-left">
    <span>홈</span><span>›</span>
    <span>국회 업무</span><span>›</span>
    <strong>{"분석 결과" if is_results else "서면질의 분석"}</strong>
  </div>
  {back_btn}
</div>
""",
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown(
        """
<div class="krds-footer">
  ⓒ 2026 기획예산처 기획조정실 | 국회 서면질의 답변서 AI 초안 생성 시스템<br>
  <small>※ AI가 생성한 초안입니다. 반드시 검토 후 사용하세요.</small>
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

def _render_input_page(openai_key: str, env_error: str | None) -> None:
    st.markdown(
        f"""
<div class="krds-input-hero">
  <h2>국회 서면질의서 분석</h2>
  <p>서면질의서 PDF를 업로드하면 AI가 질의 주제를 분류하고 답변 초안을 생성합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if env_error:
        st.error(env_error)

    st.markdown('<label class="krds-label">서면질의서 PDF 업로드</label>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "서면질의서 PDF", type=["pdf"], label_visibility="collapsed",
    )

    if uploaded_file:
        st.success(f"✓ {uploaded_file.name} 업로드 완료")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 질의서 분석 및 답변 초안 생성", type="primary",
                 use_container_width=True, disabled=env_error or not uploaded_file):
        if not uploaded_file:
            st.error("PDF 파일을 업로드하세요.")
            return

        with st.spinner("질의서를 분석 중입니다..."):
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

    # 긴급도 뱃지 색상
    urgency = result.get("urgency", "보통")
    urgency_reason = result.get("urgency_reason", "")
    urgency_class = {"높음": "krds-badge-red", "보통": "krds-badge-yellow", "낮음": "krds-badge-green"}.get(urgency, "krds-badge-yellow")

    # 질의 정보 요약
    st.markdown(
        f"""
<div class="krds-result-section" style="border-left:4px solid {PRIMARY};">
  <p class="krds-section-heading">📋 질의서 기본 정보</p>
  <table style="width:100%;border-collapse:collapse;font-size:0.875rem;">
    <tr>
      <td style="width:110px;font-weight:600;color:{GRAY_600};padding:0.3rem 0.5rem 0.3rem 0;vertical-align:top;">질의 제목</td>
      <td style="color:{GRAY_900};padding:0.3rem 0;">{_html_escape(result.get('title', '-'))}</td>
    </tr>
    <tr>
      <td style="font-weight:600;color:{GRAY_600};padding:0.3rem 0.5rem 0.3rem 0;vertical-align:top;">질의 의원</td>
      <td style="color:{GRAY_900};padding:0.3rem 0;">{_html_escape(result.get('assemblyman', '-'))}</td>
    </tr>
    <tr>
      <td style="font-weight:600;color:{GRAY_600};padding:0.3rem 0.5rem 0.3rem 0;vertical-align:top;">소속 정당</td>
      <td style="color:{GRAY_900};padding:0.3rem 0;">{_html_escape(result.get('party', '-'))}</td>
    </tr>
    <tr>
      <td style="font-weight:600;color:{GRAY_600};padding:0.3rem 0.5rem 0.3rem 0;vertical-align:top;">질문 대상 장관</td>
      <td style="color:{GRAY_900};padding:0.3rem 0;">{_html_escape(result.get('target_minister', '-'))}</td>
    </tr>
    <tr>
      <td style="font-weight:600;color:{GRAY_600};padding:0.3rem 0.5rem 0.3rem 0;vertical-align:top;">담당 부처</td>
      <td style="padding:0.3rem 0;">
        <span class="krds-badge krds-badge-blue">{_html_escape(result.get('ministry', '-'))}</span>
      </td>
    </tr>
    <tr>
      <td style="font-weight:600;color:{GRAY_600};padding:0.3rem 0.5rem 0.3rem 0;vertical-align:top;">긴급도</td>
      <td style="padding:0.3rem 0;">
        <span class="krds-badge {urgency_class}">{urgency}</span>
        {f'<span style="margin-left:0.5rem;color:{GRAY_600};font-size:0.8125rem;">{_html_escape(urgency_reason)}</span>' if urgency_reason else ''}
      </td>
    </tr>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )

    # 세부 질의 주제
    topics = result.get("topics", [])
    topics_html = "".join(
        f"""
<div class="krds-point">
  <span class="krds-point-num">{i}</span>
  <span>{_html_escape(topic)}</span>
</div>
"""
        for i, topic in enumerate(topics, 1)
    )
    st.markdown(
        f"""
<div class="krds-result-section">
  <p class="krds-section-heading">📌 세부 질의 주제</p>
  {topics_html if topics_html else f'<p style="color:{GRAY_400};">세부 주제를 파악할 수 없습니다.</p>'}
</div>
""",
        unsafe_allow_html=True,
    )

    # 질문별 카드
    questions = result.get("questions", [])
    if questions:
        cards_html = "".join(
            f"""
<div class="krds-question-card">
  <span class="krds-question-num">질문 {q.get('number', i)}</span>
  <p class="krds-question-content">{_html_escape(q.get('content', ''))}</p>
  {_render_question_answer_html(q)}
</div>
"""
            for i, q in enumerate(questions, 1)
        )
        st.markdown(
            f"""
<div class="krds-result-section">
  <p class="krds-section-heading">❓ 질문별 답변 초안</p>
  {cards_html}
</div>
""",
            unsafe_allow_html=True,
        )

    # 전체 답변서 초안
    overall_draft = result.get("overall_draft", result.get("draft_answer", ""))
    st.markdown(
        f"""
<div class="krds-result-section">
  <p class="krds-section-heading">✏️ 전체 답변서 초안</p>
  <div class="krds-draft-box">{_html_escape(overall_draft)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # 핵심 포인트
    key_points = result.get("key_points", [])
    points_html = "".join(
        f"""
<div class="krds-point">
  <span class="krds-point-num">{i}</span>
  <span>{_html_escape(point)}</span>
</div>
"""
        for i, point in enumerate(key_points, 1)
    )
    st.markdown(
        f"""
<div class="krds-result-section">
  <p class="krds-section-heading">💡 답변 핵심 포인트</p>
  {points_html if points_html else f'<p style="color:{GRAY_400};">핵심 포인트를 파악할 수 없습니다.</p>'}
</div>
""",
        unsafe_allow_html=True,
    )

    # 관련 법령
    related_laws = result.get("related_laws", [])
    laws_html = "".join(
        f"""
<div class="krds-point">
  <span class="krds-point-num">{i}</span>
  <span>{_html_escape(law)}</span>
</div>
"""
        for i, law in enumerate(related_laws, 1)
    )
    st.markdown(
        f"""
<div class="krds-result-section">
  <p class="krds-section-heading">📜 관련 법령</p>
  {laws_html if laws_html else f'<p style="color:{GRAY_400};">관련 법령을 파악할 수 없습니다.</p>'}
</div>
""",
        unsafe_allow_html=True,
    )

    # 추가 검토 사항
    additional = result.get("additional_info_needed", "")
    additional_body = (
        _html_escape(additional)
        if additional
        else f'<span style="color:{GRAY_400};">추가 검토 사항이 없습니다.</span>'
    )
    st.markdown(
        f"""
<div class="krds-result-section" style="border-left:4px solid #F6AD55;">
  <p class="krds-section-heading">⚠️ 추가 검토 사항</p>
  <p style="font-size:0.9375rem;color:{GRAY_900};line-height:1.7;">{additional_body}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # 다운로드
    doc_text = build_answer_document(result, filename)
    st.download_button(
        "💾 답변서 초안 다운로드 (.txt)",
        data=doc_text,
        file_name=f"답변서초안_{filename.replace('.pdf', '')}.txt",
        mime="text/plain",
        type="secondary",
        use_container_width=True,
    )

    # 이전으로 버튼 (숨김)
    if st.button("← 이전으로 돌아가기", type="secondary", key="btn_go_back"):
        st.session_state.page = PAGE_INPUT
        st.rerun()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="국회 서면질의 답변서 AI",
        page_icon="🏛️", layout="wide", initial_sidebar_state="collapsed",
    )
    _init_session()
    _inject_global_styles()

    openai_key = get_openai_key()
    env_error = validate_openai_key(openai_key)

    is_results = st.session_state.page == PAGE_RESULTS
    _render_site_chrome(is_results=is_results)

    if is_results:
        _render_results_page()
    else:
        _render_input_page(openai_key, env_error)

    _render_footer()


if __name__ == "__main__":
    main()