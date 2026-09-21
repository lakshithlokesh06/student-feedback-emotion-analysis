# Student Feedback Emotion Analysis

Discover the emotions behind student feedback using Natural Language Processing.

A modular Streamlit portfolio project exploring how written student feedback can help educators understand learning experiences. **Phase 2 delivers data intake, validation, cleaning, and feedback preparation. Emotion classification is not implemented yet.**

## Problem statement

Ratings and broad sentiment categories can obscure the context in student feedback. A response may express appreciation for an instructor alongside frustration with workload. This project aims to explore those nuances through emotion analysis and make future findings accessible in an educator-facing dashboard.

## Current functionality

- Four pages: Overview, Analyze Feedback, Emotion Dashboard, and About.
- Real sample row count, planned category count, and explicit “Not yet analyzed” indicators.
- Synthetic sample loading, UTF-8 CSV upload, preview, and feedback-column selection.
- Strict CSV validation with bounded file, row, and column sizes.
- Meaning-preserving feedback cleaning, row statuses, and actual data-quality metrics.
- Optional context mapping, safe date/rating conversion, and lightweight dataset profiling.
- Original, prepared, and unusable-row previews; session persistence across navigation.
- An Analyze Feedback button that explains the future implementation; it never predicts emotions.
- Empty dashboard sections reserved for future charts and contextual review.
- Reusable sample validation and pytest coverage, including Streamlit navigation.

## Planned features

Hugging Face / transformer-based emotion classification, model evaluation, emotion distributions, trends, course comparisons, recurring patterns, and feedback requiring closer review. The initial categories are joy, sadness, anger, fear, surprise, frustration, satisfaction, and neutral. These are configuration placeholders, subject to the selected model and evaluation.

## Architecture

```text
student-feedback-emotion-analysis/
├── app.py
├── .streamlit/
│   └── config.toml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── sample_data.py
│   │   ├── loader.py
│   │   ├── validation.py
│   │   ├── preparation.py
│   │   └── profiling.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── sidebar.py
│   │   ├── overview.py
│   │   ├── analysis.py
│   │   ├── intake_state.py
│   │   ├── dashboard.py
│   │   └── about.py
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_sample_data.py
│   ├── test_app.py
│   ├── test_intake.py
│   └── test_intake_ui.py
├── data/
│   └── sample_student_feedback.csv
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

`app.py` handles routing; page rendering lives in `src/ui`. `src/data` is independent of Streamlit and raises `DatasetValidationError` for expected data problems, which the UI displays. Constants and paths live in `src/config.py`; bundled paths resolve relative to the project rather than the current working directory. `loader.py` bounds and parses uploads, `validation.py` provides shared validation, `preparation.py` produces an independent prepared frame and quality counts, and `profiling.py` summarizes input data. `ui/intake_state.py` handles session transitions. Future inference can be added under `src` without mixing model logic into presentation code. No unused model abstraction or transformer dependency is introduced in this phase.

## Technology stack

Python and Streamlit power the application; pandas handles CSV data. NumPy, Plotly, and scikit-learn are included as the requested foundation for later numerical analysis, visualization, and evaluation; no charts or models use them yet. pytest is a development dependency. Native Streamlit containers, columns, and a restrained theme provide the layout without custom CSS.

## Local installation

Use Python 3.11 or newer. From the project directory, on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. Open the local URL printed by Streamlit (normally http://localhost:8501).

## Testing

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m compileall -q app.py src tests
python -m pytest -q
python -c "import app, numpy, pandas, plotly, sklearn; from src.data.sample_data import load_sample_data; print(load_sample_data().shape)"
```

Optional server smoke test:

```bash
streamlit run app.py --server.headless=true --server.address=127.0.0.1
```

In another terminal, `curl --fail http://127.0.0.1:8501/_stcore/health` should return `ok`. Stop the server with Ctrl+C. Automated UI tests also render every page and check the placeholder button and unusable-column state.

## Sample dataset

`data/sample_student_feedback.csv` contains **48 entirely synthetic, unlabeled responses** across six course/subject pairs and four semesters. Dates span January 12–April 16, 2026. Ratings range from 1 to 5. Records cover teaching satisfaction, assignment frustration, conceptual confusion, instructor appreciation, exam anxiety, practical-session excitement, disappointment, workload, positive learning experiences, neutral suggestions, and mixed feedback.

| Column | Purpose |
| --- | --- |
| `feedback_id` | Unique synthetic identifier |
| `feedback` | Written response |
| `course` | Course grouping |
| `subject` | Subject grouping |
| `semester` | Semester number |
| `rating` | Synthetic rating from 1 to 5 |
| `feedback_date` | ISO-format date |

There are no real student identities or emotion labels. Ratings are not used to infer emotions. The sample loader checks file readability, required columns, non-empty data, unique/nonblank identifiers, and usable feedback text. Uploaded CSVs may use a different schema: only a selectable text column is needed for preparation. Blank or non-text values are flagged, not silently removed. Uploads are limited to 10 MB, previewed up to 50 rows, and are not written to disk by the application.

## CSV intake and preparation

Upload a comma-separated `.csv` with one header row and at least one data row. Fields containing commas or newlines must be quoted. A custom schema is supported: explicitly select the feedback column, then optionally map course, subject, semester, rating, and feedback date. The bundled sample schema is unchanged.

Validation rules:

- Maximum **10 MiB**, **100,000 data rows**, and **100 columns**, centralized in `src/config.py`. Streamlit's transport cap is also set to 10 MB in `.streamlit/config.toml`; keep it aligned if changing the application cap.
- Reject empty files, header-only datasets, invalid UTF-8, null bytes, inconsistent field counts, malformed quoting, and oversized CSV fields (the standard CSV parser's field limit).
- Reject blank headers and duplicate names, including names that differ only by surrounding whitespace. Original header spelling is retained.
- Preserve all data records, including blank records, as well as original cell text. Do not infer null tokens or numeric types on upload. Profiling reports null counts and blank-text counts separately.

Preparation creates an independent copy with `feedback_clean`, `feedback_status`, and `feedback_is_usable`. Generated names get numbered suffixes when they collide with input columns, so existing values are never overwritten. No rows are dropped or deduplicated.

| Status | Rule |
| --- | --- |
| `valid` | Text with at least 3 characters after normalization |
| `missing` | A native null value |
| `empty` | Empty or whitespace-only text |
| `non_text` | A non-string, non-null value |
| `too_short` | Non-empty text shorter than the configured minimum |

Cleaning trims surrounding whitespace and collapses internal whitespace. It preserves case, punctuation, stopwords, and linguistic content; no stemming, lemmatization, sentiment analysis, or emotion prediction runs. The minimum length is configurable in `src/config.py`. The unusable-row view shows the original value and rejection status.

Optional date mapping creates `feedback_date_parsed` using safe ISO date/timestamp parsing in UTC. Invalid or ambiguous dates become missing parsed values while originals remain intact. Optional rating mapping creates `rating_numeric`; invalid and non-finite values become missing numeric values. Neither conversion drops rows or assumes a rating scale. Both report valid, missing/blank, and invalid counts. Ratings are not used for classification.

Profiling reports dimensions, column names, stored data types, null/blank counts, duplicate rows beyond the first occurrence, and unique selected course/subject/semester values. No charts or emotion statistics are generated.

## Session state and privacy

The active dataset, source, column mappings, prepared result, and compact profile remain in Streamlit session state across page navigation. One last valid uploaded dataset is retained for source switching, sharing the same frame reference when active. Source changes and replacement uploads reset dependent selections and preparation; navigation alone preserves them. An invalid replacement clears stale uploaded data and results. For a custom schema, choose the feedback column explicitly; known sample-style names are preselected.

The app stores no uploads on disk or in a database and makes no external model/API calls. Use anonymized feedback. Session memory is temporary, not durable storage: a new browser session or server restart can reset it. If the uploader is cleared or disappears during navigation, the last valid upload remains available until replaced or the session ends. Only the active prepared copy is retained; preparation and profiling are reused until input or mappings change.

## Roadmap

1. **Foundation (complete):** modular shell, sample data, upload preview, validation, documentation, and tests.
2. **Preparation (current):** validated CSV intake, conservative cleaning, quality reporting, context conversion, and session persistence.
3. **Classification:** select and integrate a transformer model; evaluate category compatibility and performance.
4. **Analytics:** distributions, trends, course comparisons, and contextual review.
5. **Refinement:** usability, accessibility, evaluation documentation, and portfolio presentation.

## Limitations

No emotion or sentiment classification, emotion analytics, authentication, database, or deployment is included. CSV supports comma delimiters and UTF-8 (with or without BOM); other encodings and delimiters must be converted before upload. Dates must use ISO format; ambiguous locale dates are flagged. Whitespace-only and empty feedback share the `empty` status. CSV has no native types: uploaded cells remain strings, including numeric-looking values and literal `NA`/`null`; typed non-string values from other data sources are flagged as `non_text`. Minimum length is a preparation heuristic, not proof of meaningful language. Previews are capped at 50 rows; no export is implemented. The synthetic data is for demonstrating the interface, not for model training or performance claims. Emotion predictions in later phases will require evaluation and human interpretation; text alone cannot establish a student's mental state. Dependency ranges are bounded but not a reproducible lockfile.
