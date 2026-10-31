"""Streamlit chat interface for the recruiting agent."""

import os
import uuid

import streamlit as st

import matching_agent

st.set_page_config(page_title="Agentic Profile Matching", page_icon="🎯", layout="wide")

_SCORE_THRESHOLDS = [(80, "#16a34a", "#dcfce7"), (50, "#d97706", "#fef3c7"), (0, "#dc2626", "#fee2e2")]


def _score_colors(score: float) -> tuple[str, str]:
    for threshold, fg, bg in _SCORE_THRESHOLDS:
        if score >= threshold:
            return fg, bg
    return _SCORE_THRESHOLDS[-1][1], _SCORE_THRESHOLDS[-1][2]


def _chip_html(label: str, fg: str, bg: str) -> str:
    return (
        f'<span style="display:inline-block;margin:2px 4px 2px 0;padding:2px 10px;'
        f'border-radius:999px;font-size:0.78rem;font-weight:500;'
        f'color:{fg};background:{bg};">{label}</span>'
    )


def _round_badge(round_name: str) -> str:
    labels = {
        "initial_screen": ("Initial screen", "#2563eb", "#dbeafe"),
        "deep_analysis": ("Deep analysis", "#7c3aed", "#ede9fe"),
        "final_recommendation": ("Final recommendation", "#16a34a", "#dcfce7"),
    }
    text, fg, bg = labels.get(round_name, (round_name, "#475569", "#e2e8f0"))
    return _chip_html(f"● {text}", fg, bg)


def _render_candidate_card(candidate: dict) -> None:
    fg, bg = _score_colors(candidate["score"])
    strengths = "".join(_chip_html(s, "#166534", "#f0fdf4") for s in candidate.get("strengths", [])[:4])
    gaps = "".join(_chip_html(g, "#991b1b", "#fef2f2") for g in candidate.get("gaps", [])[:4])
    st.markdown(
        f"""
        <div style="border:1px solid var(--apm-border,#e5e7eb);border-radius:14px;padding:16px 18px;
                    margin-bottom:12px;background:var(--apm-card-bg,#ffffff);
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <span style="font-weight:700;font-size:1.05rem;">{candidate['candidate_id']}</span>
            <span style="font-weight:700;font-size:1.05rem;color:{fg};background:{bg};
                         padding:3px 14px;border-radius:999px;">{candidate['score']:.0f}</span>
          </div>
          <div style="margin-bottom:6px;">{strengths or '<span style="color:#94a3b8;font-size:0.8rem;">No strengths listed</span>'}</div>
          <div>{gaps}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    :root { --apm-border: #e5e7eb; --apm-card-bg: #ffffff; }
    @media (prefers-color-scheme: dark) {
        :root { --apm-border: #334155; --apm-card-bg: #1e293b; }
    }
    .apm-hero {
        padding: 22px 28px; border-radius: 16px; margin-bottom: 18px;
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 60%, #a21caf 100%);
        color: white;
    }
    .apm-hero h1 { margin: 0 0 4px 0; font-size: 1.6rem; }
    .apm-hero p { margin: 0; opacity: 0.9; font-size: 0.95rem; }
    section[data-testid="stSidebar"] { border-right: 1px solid var(--apm-border); }
    .stButton button { border-radius: 10px; font-weight: 600; }
    </style>
    <div class="apm-hero">
        <h1>🎯 Agentic Profile Matching</h1>
        <p>LangGraph recruiting agent — multi-round screening, grounded explanations, live chat.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.job_description = ""
    st.session_state.shortlist = []
    st.session_state.round = "initial_screen"


@st.cache_resource
def _get_graph(checkpoint_path: str):
    return matching_agent.build_graph(checkpoint_path)


graph = _get_graph(os.environ.get("CHECKPOINT_PATH", "data/checkpoints.sqlite"))

with st.sidebar:
    st.markdown("### 📋 Job description")
    jd_text = st.text_area("Paste a job description", height=220, label_visibility="collapsed",
                            placeholder="Paste the full job description here…")
    if st.button("▶  Run Initial Screen", use_container_width=True):
        if not jd_text.strip():
            st.warning("Please paste a job description before running the screen.")
        else:
            with st.spinner("Parsing requirements, searching resumes, ranking candidates…"):
                result = matching_agent.run_turn(graph, st.session_state.thread_id, job_description=jd_text)
            st.session_state.job_description = jd_text
            st.session_state.shortlist = result["shortlist"]
            st.session_state.round = result["round"]
            st.session_state.chat_history.append(("assistant", result["messages"][-1].content))

    st.markdown(_round_badge(st.session_state.round), unsafe_allow_html=True)
    if st.session_state.shortlist:
        st.markdown(f"**{len(st.session_state.shortlist)}** candidates in shortlist")

chat_col, shortlist_col = st.columns([3, 2], gap="large")

with chat_col:
    st.markdown("#### 💬 Conversation")
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            if role == "assistant" and content.startswith("# Screening Report:"):
                top = st.session_state.shortlist[0] if st.session_state.shortlist else None
                summary = f"✅ Screened **{len(st.session_state.shortlist)}** candidates."
                if top:
                    summary += f" Top match: **{top['candidate_id']}** (score {top['score']:.0f})."
                summary += " Full breakdown is in the Shortlist panel →"
                st.markdown(summary)
                with st.expander("Show full report"):
                    st.markdown(content)
            else:
                st.markdown(content)

    user_message = st.chat_input("Ask about candidates...")
    if user_message:
        if not st.session_state.job_description:
            st.markdown("Please submit a job description first.")
        else:
            st.session_state.chat_history.append(("user", user_message))
            with st.spinner("Thinking…"):
                result = matching_agent.run_turn(graph, st.session_state.thread_id, user_message=user_message)
            st.session_state.shortlist = result["shortlist"]
            st.session_state.round = result["round"]
            st.session_state.chat_history.append(("assistant", result["messages"][-1].content))
            st.rerun()

with shortlist_col:
    st.markdown("#### 🏆 Shortlist")
    if not st.session_state.shortlist:
        st.caption("Candidates will appear here once a screen has run.")
    else:
        for candidate in st.session_state.shortlist:
            _render_candidate_card(candidate)
