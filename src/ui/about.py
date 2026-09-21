import streamlit as st

from src.ui.results import render_model_info


def render_about() -> None:
    st.subheader("Understanding emotion in student feedback")
    st.write("Emotion analysis is a Natural Language Processing task that aims to identify feelings expressed in text. Unlike sentiment analysis, which usually describes positive, negative, or neutral polarity, it explores more specific emotional categories.")
    with st.container(border=True):
        st.markdown("#### Educational applications")
        st.write("Potential applications include understanding responses to teaching methods, exploring workload concerns, and identifying feedback that deserves closer review. Predictions will need context and human judgment; written feedback alone cannot establish a student's mental state.")
    st.markdown("#### Current status · Phase 3")
    st.write("The application includes validated CSV intake, original and prepared previews, feedback quality summaries, optional context mapping, safe date and rating conversion, and lightweight dataset profiling. Usable feedback can now be classified locally with a pretrained English transformer. The dashboard shows real emotion and confidence distributions, and results can be exported as CSV.")
    st.markdown("#### Planned NLP capabilities")
    st.markdown("- Model evaluation and transparent reporting of limitations\n- Temporal trends and course-level exploration")
    render_model_info(st.session_state.get("intake", {}).get("analysis").metadata if st.session_state.get("intake", {}).get("analysis") else None)
    st.caption("All bundled feedback is synthetic. No student identities are included.")
