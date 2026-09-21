# Student Feedback Emotion Analysis

Discover the emotions behind student feedback using Natural Language Processing.

A modular Streamlit portfolio project exploring how written student feedback can help educators understand learning experiences. **Phase 1 delivers the project foundation and application shell. Emotion classification is not implemented.**

## Problem statement

Ratings and broad sentiment categories can obscure the context in student feedback. A response may express appreciation for an instructor alongside frustration with workload. This project aims to explore those nuances through emotion analysis and make future findings accessible in an educator-facing dashboard.

## Current functionality

- Four pages: Overview, Analyze Feedback, Emotion Dashboard, and About.
- Real sample row count, planned category count, and explicit “Not yet analyzed” indicators.
- Synthetic sample loading, UTF-8 CSV upload, preview, and feedback-column selection.
- Input checks for usable text and actionable messages for invalid files.
- An Analyze Feedback button that explains the future implementation; it never predicts emotions.
- Empty dashboard sections reserved for future charts and contextual review.
- Reusable sample validation and pytest coverage, including Streamlit navigation.

## Planned features

Text preparation, Hugging Face / transformer-based emotion classification, model evaluation, emotion distributions, trends, course comparisons, recurring patterns, and feedback requiring closer review. The initial categories are joy, sadness, anger, fear, surprise, frustration, satisfaction, and neutral. These are configuration placeholders, subject to the selected model and evaluation.

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
│   │   └── sample_data.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── sidebar.py
│   │   ├── overview.py
│   │   ├── analysis.py
│   │   ├── dashboard.py
│   │   └── about.py
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_sample_data.py
│   └── test_app.py
├── data/
│   └── sample_student_feedback.csv
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

`app.py` handles routing; page rendering lives in `src/ui`. `src/data` is independent of Streamlit and raises `DatasetValidationError` for expected data problems, which the UI displays. Constants and paths live in `src/config.py`; bundled paths resolve relative to the project rather than the current working directory. Future preparation and inference modules can be added under `src` without mixing model logic into presentation code. No unused model abstraction or transformer dependency is introduced in this phase.

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

There are no real student identities or emotion labels. Ratings are not used to infer emotions. The sample loader checks file readability, required columns, non-empty data, unique/nonblank identifiers, and usable feedback text. Uploaded CSVs may use a different schema: only a selectable text column is needed for this phase's preview. Blank or non-text values are flagged, not silently removed. Uploads are limited to 10 MB, previewed up to 50 rows, and are not written to disk by the application.

## Roadmap

1. **Foundation (current):** modular shell, sample data, upload preview, validation, documentation, and tests.
2. **Preparation:** explicit text cleaning and quality handling.
3. **Classification:** select and integrate a transformer model; evaluate category compatibility and performance.
4. **Analytics:** distributions, trends, course comparisons, and contextual review.
5. **Refinement:** usability, accessibility, evaluation documentation, and portfolio presentation.

## Limitations

No model inference, analytics, authentication, database, or deployment is included. The synthetic data is for demonstrating the interface, not for model training or performance claims. Emotion predictions in later phases will require evaluation and human interpretation; text alone cannot establish a student's mental state. Dependency ranges are bounded but not a reproducible lockfile.
