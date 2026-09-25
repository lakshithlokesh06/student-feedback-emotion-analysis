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
        st.markdown("**Phase 7 · Model monitoring**")
        st.caption("Explore the sample data and prepare feedback. Classify usable English feedback and explore model predictions.")
        st.info("Sample records are entirely synthetic.")
    return page
