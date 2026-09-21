import streamlit as st

from src.config import NAVIGATION_LABELS


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### Student feedback lab")
        st.caption("NLP · EDUCATION · INSIGHTS")
        st.divider()
        page = st.radio("Workspace", NAVIGATION_LABELS, label_visibility="collapsed")
        st.divider()
        st.caption("PROJECT STATUS")
        st.markdown("**Phase 1 · Foundation**")
        st.caption("Explore the sample data and prepare feedback. Emotion classification is planned for a later phase.")
        st.info("Sample records are entirely synthetic.")
    return page
