import streamlit as st

from src.config import MODEL_EMOTIONS
from src.ui.results import render_summary
from src.data.sample_data import DatasetValidationError, load_sample_data


def render_overview() -> None:
    st.subheader("A clearer view of the student experience")
    st.write("Written feedback captures more than a rating. This project will explore the emotions behind learning experiences to help educators understand what is working and where students need support.")
    try:
        total = str(len(load_sample_data()))
    except DatasetValidationError as exc:
        total = "Unavailable"
        st.warning(str(exc))
    result = st.session_state.get('intake', {}).get('analysis')
    if result is not None:
        render_summary(result)
    else:
        metrics = (("Total Feedback", total, "Bundled synthetic sample"),
                   ("Feedback Analyzed", "Not yet analyzed", "Run analysis to see results"),
                   ("Dominant Emotion", "Not yet analyzed", "No current predictions"),
                   ("Emotion Categories", str(len(MODEL_EMOTIONS)), "Native model categories"))
        for column, (label, value, detail) in zip(st.columns(4), metrics):
            with column, st.container(border=True):
                st.caption(label.upper())
                st.markdown(f"**{value}**")
                st.caption(detail)
    st.divider()
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("#### Beyond positive and negative")
        st.write("Sentiment analysis summarizes whether feedback is positive, negative, or neutral. Emotion analysis aims to identify more specific feelings, such as satisfaction, frustration, or fear.")
    with right, st.container(border=True):
        st.markdown("#### Context matters")
        st.write("A student may enjoy practical sessions while feeling anxious about an exam. Future analysis will help surface these nuances, while preserving the original feedback for human interpretation.")
    st.subheader("From feedback to insight")
    st.caption("Planned workflow · Data preparation, classification, and basic distributions are available in Phase 3.")
    steps = (
        ("Student Feedback", "Upload a CSV or explore the synthetic sample.", "Available"),
        ("Text Preparation", "Validate and prepare written responses.", "Available"),
        ("Emotion Classification", "Identify emotions with an NLP model.", "Available"),
        ("Emotion Analytics", "Explore emotion and confidence distributions.", "Available"),
        ("Actionable Insights", "Review findings to inform educational improvements.", "Planned"),
    )
    for index, (title, detail, status) in enumerate(steps, 1):
        with st.container(border=True):
            st.markdown(f"**{index:02d} · {title}**")
            st.caption(f"{detail} · {status}")
