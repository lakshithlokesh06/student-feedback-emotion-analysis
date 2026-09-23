# Student Feedback Emotion Analysis

Discover the emotions behind student feedback using Natural Language Processing.

A modular Streamlit portfolio project exploring how written student feedback can help educators understand learning experiences. **Phase 4 adds advanced descriptive analytics, shared dashboard filters, contextual review, and deterministic insights on top of real pretrained emotion classification.**

## Problem statement

Ratings and broad sentiment categories can obscure the context in student feedback. A response may express appreciation for an instructor alongside frustration with workload. This project aims to explore those nuances through emotion analysis and make future findings accessible in an educator-facing dashboard.

## Current functionality

- Four pages: Overview, Analyze Feedback, Emotion Dashboard, and About.
- Real result metrics after analysis and explicit “Not yet analyzed” indicators before it.
- Synthetic sample loading, UTF-8 CSV upload, preview, and feedback-column selection.
- Strict CSV validation with bounded file, row, and column sizes.
- Meaning-preserving feedback cleaning, row statuses, and actual data-quality metrics.
- Optional context mapping, safe date/rating conversion, and lightweight dataset profiling.
- Original, prepared, and unusable-row previews; session persistence across navigation.
- Explicit Analyze Feedback action with lazy model loading, batched CPU inference, and progress reporting.
- Emotion counts/percentages, confidence distribution, dominant emotion, filtering, and CSV export.
- Reusable sample validation and pytest coverage, including Streamlit navigation.

## Planned features

Domain-specific model evaluation, confidence calibration, and further usability refinement. Frustration and satisfaction remain future category goals, not current predictions.

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
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── distributions.py
│   │   ├── comparisons.py
│   │   ├── trends.py
│   │   ├── filters.py
│   │   └── insights.py
│   ├── emotion/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   ├── labels.py
│   │   └── model_loader.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── sidebar.py
│   │   ├── overview.py
│   │   ├── analysis.py
│   │   ├── intake_state.py
│   │   ├── model_cache.py
│   │   ├── results.py
│   │   ├── dashboard.py
│   │   ├── dashboard_filters.py
│   │   ├── dashboard_sections.py
│   │   └── about.py
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_sample_data.py
│   ├── test_app.py
│   ├── test_intake.py
│   ├── test_intake_ui.py
│   ├── test_emotion.py
│   ├── test_analytics.py
│   └── test_dashboard.py
├── data/
│   └── sample_student_feedback.csv
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

`app.py` handles routing; page rendering lives in `src/ui`. `src/data` is independent of Streamlit and raises `DatasetValidationError` for expected data problems, which the UI displays. Constants and paths live in `src/config.py`; bundled paths resolve relative to the project rather than the current working directory. `loader.py` bounds and parses uploads, `validation.py` provides shared validation, `preparation.py` produces an independent prepared frame and quality counts, and `profiling.py` summarizes input data. `ui/intake_state.py` handles session transitions. `src/emotion` owns model loading, metadata-driven labels, batched classification, summaries, and flat CSV export. Streamlit resource caching stays in `ui/model_cache.py`; model logic does not depend on Streamlit.

## Technology stack

Python and Streamlit power the application; pandas handles CSV data. NumPy supports preparation and Plotly powers result charts. Transformers and PyTorch provide pretrained CPU inference. Scikit-learn remains available for future evaluation. pytest is a development dependency. Native Streamlit containers, columns, and a restrained theme provide the layout without custom CSS.

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

In another terminal, `curl --fail http://127.0.0.1:8501/_stcore/health` should return `ok`. Stop the server with Ctrl+C. Automated UI tests render every page, exercise successful inference with an explicitly mocked model, cover loading failures and invalidation, and check unusable-column states. Unit tests never download weights. The original placeholder-button test now mocks a model-loading failure to preserve its no-predictions error-path assertion.

Optional real inference test (downloads weights on first run):

```bash
RUN_MODEL_INTEGRATION=1 python -m pytest tests/test_emotion.py::test_real_model_smoke -q -s
python -m pip check
```

This integration check includes an overlength response to verify truncation. It checks valid outputs, not model accuracy.

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

Cleaning trims surrounding whitespace and collapses internal whitespace. It preserves case, punctuation, stopwords, and linguistic content; no stemming, lemmatization, or sentiment analysis runs. Emotion inference is a separate explicit action after preparation. The minimum length is configurable in `src/config.py`. The unusable-row view shows the original value and rejection status.

Optional date mapping creates `feedback_date_parsed` using safe ISO date/timestamp parsing in UTC. Invalid or ambiguous dates become missing parsed values while originals remain intact. Optional rating mapping creates `rating_numeric`; invalid and non-finite values become missing numeric values. Neither conversion drops rows or assumes a rating scale. Both report valid, missing/blank, and invalid counts. Ratings are not used for classification.

Profiling reports dimensions, column names, stored data types, null/blank counts, duplicate rows beyond the first occurrence, and unique selected course/subject/semester values. Profiling itself does not generate emotion statistics; those come only from successful classification.

## Session state and privacy

The active dataset, source, column mappings, prepared result, and compact profile remain in Streamlit session state across page navigation. One last valid uploaded dataset is retained for source switching, sharing the same frame reference when active. Source changes and replacement uploads reset dependent selections and preparation; navigation alone preserves them. An invalid replacement clears stale uploaded data and results. For a custom schema, choose the feedback column explicitly; known sample-style names are preselected.

The app stores no uploads on disk or in a database. Model files are downloaded from Hugging Face; feedback is processed locally, not sent to a hosted inference API. Use anonymized feedback. Session memory is temporary, not durable storage: a new browser session or server restart can reset it. If the uploader is cleared or disappears during navigation, the last valid upload remains available until replaced or the session ends. The active prepared copy and completed analysis are retained; preparation and profiling are reused until input or mappings change. Dataset, feedback-column, and context-mapping changes invalidate emotion results. Navigation and result filtering never trigger inference. A failed analysis saves no partial predictions and can be retried explicitly.

## Emotion classification

The selected model is [Jochen Hartmann’s emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base), a distilled English RoBERTa model fine-tuned for emotion classification. Its compact architecture and seven native emotion labels fit this CPU-first foundation. Supported labels are **anger, disgust, fear, joy, neutral, sadness, and surprise**. The model is pinned to revision `0e1cd914e3d46199ed785853e12b57304e04178b`. Model metadata determines the label order; generic `LABEL_n` values are resolved only through that metadata. No undocumented label mapping is used.

Frustration and satisfaction are not native labels. Anger is not automatically frustration, and joy is not automatically satisfaction; the app does not derive either category. The earlier future-category configuration remains a roadmap placeholder.

Click **Analyze Feedback** after preparation. The app loads the model only then, reuses it through Streamlit resource caching, and classifies only usable rows in batches of **8** on **CPU**. Inference calls are protected by a lock for the shared model. Inputs are padded and truncated to at most **512 tokens**, including special tokens. Original text is never truncated or overwritten. Large datasets can take considerable time; start with the sample or a small upload. A normal laptop needs sufficient RAM for PyTorch, the model, and the data; no GPU is required. First use requires internet access and several hundred MB of download/cache space. Weights are cached in the ignored project-local `.model_cache` directory. Later runs reuse downloaded files. Install dependencies into the virtual environment before running; no model loads during ordinary startup/navigation.

Each analyzed row receives `emotion_label`, numeric `emotion_confidence` in [0, 1], `emotion_status=analyzed`, and `emotion_low_confidence`. Rejected rows remain `not_analyzed` with blank labels/confidence/low-confidence flags. Colliding input column names receive numbered output suffixes, preserving the original fields.

Confidence is the softmax probability of the highest-scoring class, not a calibrated probability that the prediction is correct. The UI rounds it to a percentage while CSV retains the numeric value. Scores strictly below **0.50** are flagged but never dropped; change `LOW_CONFIDENCE_THRESHOLD` in configuration to adjust future runs. Results store the threshold and loaded model revision used for that run. No fabricated scores or substitute heuristic predictions are used when the model fails.

The dashboard shows emotion counts and percentages, a confidence histogram, and dominant emotion (all tied labels when applicable). Percentages and average confidence use analyzed rows only. Download exports every original row and column, preparation fields, and flat emotion outputs as UTF-8 CSV, regardless of the preview filter. It never exports the model or internal Python objects.

These probabilistic predictions are for educational and analytical use, not ground truth. The model is English-only, chooses one dominant label per response, can misread sarcasm or mixed emotions, and may lose context through truncation. No student-feedback benchmark, fairness study, or confidence calibration has been completed for this project. Handle feedback responsibly, use anonymized inputs, and review outputs in context.

## Advanced dashboard (Phase 4)

The dashboard operates exclusively on the stored Phase 3 analyzed result; no model runs when a filter or dashboard section changes. Analytics use a narrow copied view with positional row references, preserving every original value and prediction even when source row indices or column names collide. The model, its labels, and its confidence scores are unchanged.

- **Overview:** filtered total rows, analyzed rows, dominant emotion (including ties), average confidence, low-confidence count, detected categories, emotion counts/percentages, and confidence histogram. Native emotion order is consistent and is not a severity ranking.
- **Course / Subject / Semester:** descriptive counts, dominant emotion, average confidence, stacked emotion counts, and percentage tables. Semester labels may be text. Displays up to 15 groups by response volume with deterministic tie handling; filter to inspect other groups. Missing/blank group context is excluded from that section and reported. Groups are not rated best/worst.
- **Rating vs Emotion:** average numeric rating by emotion, feedback counts, and emotion composition by rating. Displays up to 20 rating values by volume; filters can select others. Missing/invalid/nonfinite ratings are excluded only here and counted. No rating scale or causal relationship is assumed.
- **Trends Over Time:** daily, Monday-start weekly, or monthly aggregation in UTC. Default is daily for up to 31 days, weekly for up to 180 days, otherwise monthly. Invalid/missing dates are counted and excluded only from temporal aggregation. Charts show feedback volume and emotion counts or percentages. Empty periods have zero counts and undefined percentages. More than 600 periods requires coarser aggregation or narrower filters.
- **Confidence Analysis:** average/median confidence, low-confidence count/share using the analyzed run's threshold, and per-emotion confidence summary and box plot. Confidence is not validated accuracy.
- **Feedback Review:** analyzed responses with selected context, filtered through the shared controls and sorted by confidence. Preview is capped at 50 rows.
- **Feedback Requiring Attention:** “Feedback that may benefit from closer review” selects anger, fear, sadness, and disgust predictions, sorted by confidence. Low-confidence predictions are retained. This is a descriptive human-review aid, not a diagnosis, danger assessment, or disciplinary recommendation.
- **Insights:** deterministic observations of dominant-emotion share, low-confidence share, and highest mean model confidence, including ties. No LLM, fabricated statistics, or causal explanations are used. All observations explicitly refer to the current filtered view.

Dashboard filters support emotion, course, subject, semester, numeric rating values, and confidence level when fields are available. Selections combine with AND; empty selections include all values, including missing context. Confidence filters include analyzed rows only. Reset Filters restores all rows. Filter state lives separately from inference state, survives navigation, and resets on new/invalidated analysis. Every dashboard metric, chart, insight, review, and filtered export uses the same filtered view; optional sections further exclude only rows missing their required context. Missing fields, zero analyzed rows, and zero-match filters have explicit empty states.

**Download full analyzed dataset** retains the full existing export. **Download current filtered view** exports all matching rows (not only the preview) and all original/preparation/emotion fields as UTF-8 CSV. No model objects or chart images are exported.

The dashboard describes model outputs and associations, not causation or ground truth. Sparse groups/periods may give unstable proportions. Human review is required before decisions based on individual feedback. The system does not diagnose student wellbeing or mental-health conditions and does not infer characteristics about individual students.

## Roadmap

1. **Foundation (complete):** modular shell, sample data, upload preview, validation, documentation, and tests.
2. **Preparation (complete):** validated CSV intake, conservative cleaning, quality reporting, context conversion, and session persistence.
3. **Classification (complete):** pretrained CPU inference, confidence reporting, basic distributions, and CSV export. Domain-specific evaluation is still planned.
4. **Advanced analytics (current):** context comparisons, rating associations, temporal trends, confidence exploration, filters, review aids, and deterministic insights.
5. **Refinement:** usability, accessibility, evaluation documentation, and portfolio presentation.

## Limitations

No sentiment analysis, authentication, database, or deployment is included. CSV supports comma delimiters and UTF-8 (with or without BOM); other encodings and delimiters must be converted before upload. Dates must use ISO format; ambiguous locale dates are flagged. Whitespace-only and empty feedback share the `empty` status. CSV has no native types: uploaded cells remain strings, including numeric-looking values and literal `NA`/`null`; typed non-string values from other data sources are flagged as `non_text`. Minimum length is a preparation heuristic, not proof of meaningful language. Previews are capped at 50 rows; export includes all rows. The synthetic data is for demonstrating the interface, not for model training or performance claims. Emotion predictions require evaluation and human interpretation; text alone cannot establish a student's mental state. Dependency ranges are bounded but not a reproducible lockfile.
