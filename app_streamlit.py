"""Streamlit chat interface for the recruiting agent."""

import os
import uuid

import streamlit as st

import matching_agent

st.title("Agentic Profile Matching")

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
    st.subheader("Job description")
    jd_text = st.text_area("Paste a job description", height=200)
    if st.button("Run Initial Screen"):
        result = matching_agent.run_turn(graph, st.session_state.thread_id, job_description=jd_text)
        st.session_state.job_description = jd_text
        st.session_state.shortlist = result["shortlist"]
        st.session_state.round = result["round"]
        st.session_state.chat_history.append(("assistant", result["messages"][-1].content))

    st.caption(f"Round: {st.session_state.round}")

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

if st.session_state.shortlist:
    with st.expander(f"Shortlist ({len(st.session_state.shortlist)} candidates)"):
        for candidate in st.session_state.shortlist:
            st.markdown(f"**{candidate['candidate_id']}** — score {candidate['score']}")
            st.markdown(f"Strengths: {', '.join(candidate.get('strengths', []))}")
            st.markdown(f"Gaps: {', '.join(candidate.get('gaps', []))}")

user_message = st.chat_input("Ask about candidates...")
if user_message:
    if not st.session_state.job_description:
        st.markdown("Please submit a job description first.")
    else:
        st.session_state.chat_history.append(("user", user_message))
        result = matching_agent.run_turn(graph, st.session_state.thread_id, user_message=user_message)
        st.session_state.shortlist = result["shortlist"]
        st.session_state.chat_history.append(("assistant", result["messages"][-1].content))
        st.rerun()
