import streamlit as st

from src.config import FUTURE_EMOTION_CATEGORIES
from src.data.sample_data import DatasetValidationError, load_sample_data


def render_overview() -> None:
    st.subheader("A clearer view of the student experience")
    st.write("Written feedback captures more than a rating. This project will explore the emotions behind learning experiences to help educators understand what is working and where students need support.")
    try:
        total = str(len(load_sample_data()))
    except DatasetValidationError as exc:
        total = "Unavailable"
        st.warning(str(exc))
    metrics = (
        ("Total Feedback", total, "Bundled synthetic sample"),
        ("Feedback Analyzed", "Not yet analyzed", "Classification arrives in a later phase"),
        ("Dominant Emotion", "Not yet analyzed", "No predictions have been generated"),
        ("Emotion Categories", str(len(FUTURE_EMOTION_CATEGORIES)), "Planned categories · configuration only"),
    )
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
    st.caption("Planned workflow · Only data loading and preview are available in Phase 1.")
    steps = (
        ("Student Feedback", "Upload a CSV or explore the synthetic sample.", "Available"),
        ("Text Preparation", "Validate and prepare written responses.", "Planned"),
        ("Emotion Classification", "Identify emotions with an NLP model.", "Planned"),
        ("Emotion Analytics", "Explore distributions, trends, and course patterns.", "Planned"),
        ("Actionable Insights", "Review findings to inform educational improvements.", "Planned"),
    )
    for index, (title, detail, status) in enumerate(steps, 1):
        with st.container(border=True):
            st.markdown(f"**{index:02d} · {title}**")
            st.caption(f"{detail} · {status}")
