"""Streamlit entry point: presentation routing only; no model inference."""
import streamlit as st

from src.config import APP_SUBTITLE, APP_TITLE
from src.ui.about import render_about
from src.ui.evaluation import render_evaluation
from src.ui.analysis import render_analysis
from src.ui.dashboard import render_dashboard
from src.ui.overview import render_overview
from src.ui.sidebar import render_sidebar


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📘", layout="wide")
    st.caption("STUDENT FEEDBACK LAB / PHASE 5")
    st.title(APP_TITLE)
    st.write(APP_SUBTITLE)
    st.divider()
    page = render_sidebar()
    {
        "Overview": render_overview,
        "Analyze Feedback": render_analysis,
        "Emotion Dashboard": render_dashboard,
        "About": render_about,
        "Model Evaluation": render_evaluation,
    }[page]()
    st.divider()
    st.caption("Student Feedback Emotion Analysis · Portfolio project · Model evaluation release")


if __name__ == "__main__":
    main()
