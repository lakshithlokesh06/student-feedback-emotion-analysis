import streamlit as st

from src.config import FUTURE_EMOTION_CATEGORIES


def render_about() -> None:
    st.subheader("Understanding emotion in student feedback")
    st.write("Emotion analysis is a Natural Language Processing task that aims to identify feelings expressed in text. Unlike sentiment analysis, which usually describes positive, negative, or neutral polarity, it explores more specific emotional categories.")
    with st.container(border=True):
        st.markdown("#### Educational applications")
        st.write("Potential applications include understanding responses to teaching methods, exploring workload concerns, and identifying feedback that deserves closer review. Predictions will need context and human judgment; written feedback alone cannot establish a student's mental state.")
    st.markdown("#### Current status · Phase 1")
    st.write("The foundation includes modular navigation, a validated synthetic dataset, CSV upload and preview, text-column selection, and dashboard placeholders. There is no emotion model, inference, or analytical output yet.")
    st.markdown("#### Planned NLP capabilities")
    st.markdown("- Text preparation and input quality checks\n- Transformer-based emotion classification using Hugging Face\n- Model evaluation and transparent reporting of limitations\n- Emotion distributions, temporal trends, and course-level exploration")
    st.caption("INITIAL CATEGORY CONFIGURATION · SUBJECT TO MODEL EVALUATION")
    st.write(" · ".join(FUTURE_EMOTION_CATEGORIES))
    st.caption("All bundled feedback is synthetic. No student identities are included.")
