import pandas as pd
import streamlit as st

from src.data.sample_data import DatasetValidationError, load_sample_data


def render_analysis() -> None:
    st.subheader("Prepare your feedback")
    st.write("Choose a dataset, inspect its contents, and select the column containing written feedback.")
    st.info("Phase 1 supports data preview only. Emotion analysis will be implemented in a later phase.")
    source = st.radio("Data source", ("Sample dataset", "Upload CSV"), horizontal=True)
    data = None
    if source == "Sample dataset":
        st.caption("48 synthetic student responses · No real student information or emotion labels.")
        try:
            data = load_sample_data()
        except DatasetValidationError as exc:
            st.error(str(exc))
    else:
        uploaded = st.file_uploader("Upload a UTF-8 CSV", type=["csv"], help="Maximum 10 MB. Include at least one column with written feedback; the sample schema is optional.")
        st.caption("Use anonymized feedback. Uploaded files are processed in this session and are not saved by the application.")
        if uploaded is not None:
            try:
                data = pd.read_csv(uploaded)
                if data.empty:
                    st.error("This CSV has no feedback rows. Upload a file with at least one row.")
                    data = None
            except (UnicodeError, ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                st.error("Could not read this file. Upload a valid UTF-8 CSV with a header row and consistent columns.")
    if data is None:
        st.button("Analyze Feedback", disabled=True, type="primary")
        return
    st.markdown("#### Dataset preview")
    st.caption(f"{len(data):,} rows · {len(data.columns)} columns · Showing up to 50 rows")
    st.dataframe(data.head(50), hide_index=True, width="stretch")
    columns = list(data.columns)
    selected = st.selectbox("Feedback text column", columns, index=columns.index("feedback") if "feedback" in columns else 0)
    usable = data[selected].map(lambda value: isinstance(value, str) and bool(value.strip()))
    usable_count = int(usable.sum())
    st.caption(f"{usable_count:,} of {len(data):,} rows contain usable text in this column.")
    if not usable.all():
        st.warning("The selected column contains missing, blank, or non-text values. Choose a text column or clean these rows before future analysis.")
    if st.button("Analyze Feedback", type="primary", disabled=usable_count == 0):
        st.info("Emotion analysis will be implemented in a later phase. No predictions have been generated or saved.")
