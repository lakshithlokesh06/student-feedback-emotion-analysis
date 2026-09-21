import streamlit as st


def render_dashboard() -> None:
    st.subheader("Emotion Dashboard")
    st.info("No feedback has been analyzed yet. Charts and insights will appear here after emotion classification is implemented and feedback is analyzed.")
    areas = (
        ("Emotion distribution", "The relative frequency of identified emotions across feedback."),
        ("Emotion trends", "Changes in emotion patterns over time, when dates are available."),
        ("Emotion by course / subject", "Course and subject comparisons, when those fields are available."),
        ("Most common emotional patterns", "Recurring themes and combinations for further exploration."),
        ("High-concern feedback", "Feedback that may warrant contextual review by an educator."),
    )
    columns = st.columns(2)
    for index, (title, description) in enumerate(areas):
        with columns[index % 2], st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(description)
            st.caption("AWAITING ANALYSIS")
