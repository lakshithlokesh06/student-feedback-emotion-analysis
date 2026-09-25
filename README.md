# Student Feedback Emotion Analysis

Discover the emotions behind student feedback using Natural Language Processing.

A modular Streamlit portfolio project exploring how written student feedback can help educators understand learning experiences. **Phase 6 adds human review, annotation provenance, agreement analysis, and curated labeled dataset exports.**

## Problem statement

Ratings and broad sentiment categories can obscure the context in student feedback. A response may express appreciation for an instructor alongside frustration with workload. This project aims to explore those nuances through emotion analysis and make future findings accessible in an educator-facing dashboard.

## Current functionality

- Five pages: Overview, Analyze Feedback, Emotion Dashboard, Model Evaluation, and About.
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
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   ├── confidence.py
│   │   ├── errors.py
│   │   ├── quality.py
│   │   └── state.py
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
│   │   ├── evaluation.py
│   │   ├── evaluation_sections.py
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
│   ├── test_dashboard.py
│   └── test_evaluation.py
├── data/
│   ├── sample_student_feedback.csv
│   └── sample_labeled_feedback.csv
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

These probabilistic predictions are for educational and analytical use, not ground truth. The model is English-only, chooses one dominant label per response, can misread sarcasm or mixed emotions, and may lose context through truncation. The evaluation workflow is implemented, but no representative human-annotated benchmark, fairness study, or validated confidence calibration has been completed for this project. Handle feedback responsibly, use anonymized inputs, and review outputs in context.

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

## Model Evaluation (Phase 5)

**Model Evaluation** is a separate workspace for comparing model outputs with supplied reference labels. Upload UTF-8 CSV with required columns `feedback` and `true_emotion`; optional context is preserved without being required. Existing file, row, column, encoding, and header validation applies. Ground-truth labels are trimmed and lowercased and must be one of anger, disgust, fear, joy, neutral, sadness, or surprise. Frustration, satisfaction, positive, negative, and other unrelated labels are rejected, never mapped.

The original label and text are retained. The normalized label, validation reason, and duplicate flag are stored in added fields with collision-safe names. Rows with missing/unsupported labels or unusable text are preserved but excluded from inference and all metrics. Duplicated normalized feedback/label pairs are retained and reported, including pairs with differing context; repeated examples may inflate apparent evidence. A valid schema with some rejected rows can still be evaluated; no valid rows disables the run action.

### Synthetic labeled sample

`data/sample_labeled_feedback.csv` contains **49 new synthetic examples, seven per native emotion**, with course, subject, and semester context. Labels were authored for this demonstration, not independently adjudicated by human annotators. It contains no real student information and does not copy the original sample row-for-row. **This is a workflow demonstration, not a validated research benchmark. Synthetic-data results do not establish real-world or production accuracy.** Select the bundled sample and click Run Evaluation to explore the workflow.

### Inference and state

Evaluation reuses the same cached model, pinned revision, eight-row CPU batching, tokenizer, normalization, and 512-token limit. Only its optional detailed-output path exposes the existing softmax vector; the normal inference label/score contract and main table remain unchanged. Evaluation does not train, fine-tune, calibrate, or load a second model.

Evaluation dataset, source, results, review filters, and status live under a separate session-state namespace. Navigation preserves current data/results and error filters. Replacing evaluation data or switching sources invalidates only evaluation results and filters. Changing the normal analysis dataset leaves evaluation untouched. Evaluation uploads remain in session memory across navigation; source switching clears them. Inference runs only on Run Evaluation, never on a review-filter change. Failures save no partial results.

### Metrics and confusion matrix

Accuracy, macro and weighted precision/recall/F1, and per-class precision/recall/F1/support are computed with scikit-learn from successfully evaluated rows only. Macro metrics average **all seven native classes**, with undefined class scores set to zero, even when classes are absent; weighted metrics use true-label support. No evaluated rows yields unavailable aggregate metrics rather than claimed zero accuracy. Inspect class support when interpreting averages.

The confusion matrix keeps the native label order, true labels on rows and predictions on columns. Toggle raw counts or row-normalized percentages. Empty true-label rows display zero and have no recall evidence. True/predicted class distributions report counts and percentages without causal interpretations.

### Errors, confidence, and reliability

Error review filters by true label, predicted label, minimum confidence, and correctness. Separate tables show high-confidence misclassifications (default score ≥ 0.85) and ambiguous predictions. Review filters affect these tables, not the full-run headline metrics or exports. Every preview is bounded to 50 rows.

Confidence analysis compares mean confidence and distributions for correct/incorrect predictions, plus accuracy by bands `[0, .50)`, `[.50, .70)`, `[.70, .85)`, and `[.85, 1]`. Band edges are configurable. A ten-bin reliability chart compares mean confidence with observed accuracy and reports support; empty bins have no accuracy estimate. Warnings cover fewer than 30 evaluated examples, fewer than 3 examples for a represented class, one/two represented labels, a class occupying at least 70% of labels, duplicates, synthetic data, and especially unstable calibration below 100 examples. No recalibration is performed.

Confidence is not correctness or validated accuracy. High-confidence predictions can be wrong. Domain-specific, independently labeled data is required for meaningful assessment, and the tool is not validated for educational decision-making.

### Score transparency and truncation

For a selected evaluated row, inspect the original feedback, normalized reference label, predicted label, complete seven-class probabilities, second-highest label/score, and top-two margin. A margin strictly below 0.10 sets `ambiguous_prediction`; thresholds are in `src/config.py`. This flag is analytical and does not claim that the model understands or fails to understand the text.

Token length is measured on prepared feedback before truncation, including special tokens. Evaluation records truncation status and reports count/share. Accuracy by truncated/non-truncated groups is displayed only when each group has at least three examples; it does not imply truncation caused errors. Original feedback is never truncated in storage/export.

Explainability here means **model-score transparency**, not attribution. Word/token importance, SHAP, LIME, and other attribution frameworks are intentionally omitted because scores alone cannot establish why a model chose a label. No fabricated explanations or LLM-generated interpretations are used.

### Evaluation exports and tests

Download all evaluation rows as UTF-8 CSV, per-class metrics as CSV, or only misclassified evaluated rows as CSV. Results include original feedback/labels/context, preparation/validation fields, normalized truth, prediction, confidence, correctness, low-confidence/ambiguity flags, runner-up scores, flat class probabilities, and truncation fields. Rejected rows have no prediction or correctness value. Exports never contain model instances or serialized Python objects.

Offline tests use test doubles and cover validation, metrics, absent/empty classes, confusion, confidence bins, error selection, transparency fields, exports, and independent session state. The real-model evaluation test is opt-in:

```bash
RUN_MODEL_INTEGRATION=1 python -m pytest tests/test_evaluation.py::test_real_evaluation -q -s
```

## Human Review (Phase 6)

Human review uses existing predictions from **Main Analysis Results** or **Evaluation Results**, in separate workspaces. Analyze the bundled sample (or upload and analyze feedback), open Human Review, select a source, and inspect one item at a time. Choose a decision, optionally add a note and local reviewer ID, then use **Save Review** or **Save & Next**. Previous, Next, numbered jump, First unreviewed item, and Clear current item selection support navigation. Save edits before navigating.

| Status | Meaning |
| --- | --- |
| unreviewed | No saved decision |
| accepted | Reviewer agrees with the original predicted emotion |
| corrected | Reviewer selects a different supported emotion |
| uncertain | Reviewer cannot confidently assign one emotion; reviewed label stays empty |
| skipped | Review is deliberately deferred; reviewed label stays empty |

The seven supported labels remain anger, disgust, fear, joy, neutral, sadness, and surprise. Notes are optional, trimmed and normalized, with control characters removed and a 500-character maximum. The optional reviewer ID defaults to `local_reviewer`; it is provenance metadata, not authentication. Use a non-identifying label if possible.

### Queues, progress, and state

Priorities include all predictions, low-confidence predictions, high-confidence predictions (≥0.85), a specific emotion, and feedback requiring attention (anger, fear, sadness, disgust). Evaluation also supports ambiguous predictions and misclassified rows. Optional detail controls appear only when those fields exist; normal analysis does not rerun inference to obtain runner-up or ambiguity information. These modes select and order NLP outputs; they do not rank students or infer risk.

Filters cover review status, model label, reviewed label, confidence range, low-confidence flag, ambiguity, and available course/subject/semester context. Reset Filters restores the full queue and default priority without deleting annotations. Progress and exports always use the full source, regardless of filters. Reviewed = accepted + corrected + uncertain; Remaining = unreviewed + skipped. Uncertainty is a legitimate completed review outcome; skipped items remain deferred.

Annotations, queue preferences, and selected items live in Streamlit session state. Navigation and dashboard filtering preserve saved reviews. Dataset replacement, preparation changes, or a new analysis/evaluation run invalidate the affected source's previous review snapshot when Human Review next opens. The other source is unaffected. **Clear ALL saved review progress for this source** explicitly removes that source's decisions and audit events. Export before clearing or ending a session; there is no persistent storage.

Review IDs are deterministic SHA-256 hashes incorporating source, dataset identity, optional original feedback ID, feedback text, row content, and duplicate occurrence. Dataset identity includes columns, row content/order, source, and model revision. Duplicate rows receive distinct IDs; IDs remain stable through filters and navigation. Reordering or changing the dataset creates a new snapshot. Displayed IDs expose only a hash prefix. Restore/import is future work and would require snapshot validation and explicit conflict handling.

### Agreement and correction patterns

Model-human agreement is accepted / (accepted + corrected). Uncertain/skipped decisions are excluded from this denominator. This is agreement with the session's annotations, **not objective accuracy**. Counts, correction transitions, corrections by predicted/reviewed emotion, average confidence for accepted/corrected predictions, and correction rates by confidence band describe only the reviewed subset. The seven-class agreement matrix appears after three accepted/corrected decisions. Warnings identify fewer than ten accepted/corrected decisions or fewer than three decisions for a represented predicted emotion.

For evaluation data, annotation comparison reports all three agree, model/human agree, reference/human agree, model/reference agree, or all differ. No label source is treated as infallible. Observations are deterministic and descriptive; they do not establish causation.

### Annotation exports and integrity

Three UTF-8 CSV downloads are available:

- **Reviewed Feedback CSV:** all saved statuses, original feedback/prediction/confidence, available context/details, stable ID, source/snapshot identity, reviewed label, note, timestamp, and reviewer ID.
- **Curated Labeled Dataset CSV:** accepted/corrected rows only, with `feedback`, human-reviewed label as `true_emotion`, and available context. Uncertain and skipped rows are excluded. This schema can be uploaded to Model Evaluation.
- **Annotation Audit CSV:** every save event, including edits, original label/confidence, reviewed label/status/note, UTC timestamp, reviewer ID, source, and snapshot identity.

Latest decisions are stored separately from the append-only save-event audit. Review never modifies original feedback, model predictions/confidence, main analysis, or evaluation results. Changing reviewer ID does not create an independent annotation panel: this version retains one latest decision per row and an audit of edits, ready for future multiple-reviewer support.

Reviewer labels are human annotations, not guaranteed ground truth, and reviewers may disagree. Single-label classification can oversimplify mixed emotions: select the dominant emotion when reasonable or mark uncertain. Reviewed datasets can support future evaluation or fine-tuning, but **the model is unchanged and annotation never retrains or fine-tunes it**. There are no accounts, adjudication, inter-rater statistics, import/restore, or persistent storage in this phase.

Review logic lives in `src/review/` (identifiers, queue, validation, state, metrics, export); Streamlit rendering lives in `src/ui/review.py` and `src/ui/review_sections.py`. Offline fixtures and AppTest checks cover review outcomes, immutability, identifiers, filters/priorities, metrics, exports, navigation, and source invalidation without transformer downloads.

## Roadmap

1. **Foundation (complete):** modular shell, sample data, upload preview, validation, documentation, and tests.
2. **Preparation (complete):** validated CSV intake, conservative cleaning, quality reporting, context conversion, and session persistence.
3. **Classification (complete):** pretrained CPU inference, confidence reporting, basic distributions, and CSV export. Domain-specific evaluation is still planned.
4. **Advanced analytics (complete):** context comparisons, rating associations, temporal trends, confidence exploration, filters, review aids, and deterministic insights.
5. **Evaluation (complete):** labeled-data validation, metrics, errors, confidence reliability, score transparency, and exports.
6. **Human review (current):** annotation queues, decisions, agreement, provenance, and curated dataset exports.
7. **Refinement (planned):** representative human-annotated assessment, usability, accessibility, and portfolio presentation.

## Limitations

No sentiment analysis, authentication, database, or deployment is included. CSV supports comma delimiters and UTF-8 (with or without BOM); other encodings and delimiters must be converted before upload. Dates must use ISO format; ambiguous locale dates are flagged. Whitespace-only and empty feedback share the `empty` status. CSV has no native types: uploaded cells remain strings, including numeric-looking values and literal `NA`/`null`; typed non-string values from other data sources are flagged as `non_text`. Minimum length is a preparation heuristic, not proof of meaningful language. Previews are capped at 50 rows; export includes all rows. The synthetic data is for demonstrating the interface, not for model training or performance claims. Emotion predictions require evaluation and human interpretation; text alone cannot establish a student's mental state. Dependency ranges are bounded but not a reproducible lockfile.

## Phase 7 — Model Monitoring

**Model Monitoring** compares a reference dataset with current analyzed feedback:
monitoring identifies changes, **Model Evaluation** measures performance against
labels, and **Human Review** captures human annotations. The existing model is
unchanged; this phase does not retrain it or add production monitoring infrastructure.

1. Classify the bundled sample in **Analyze Feedback**.
2. Open **Model Monitoring**, then **Set main analysis as Reference Snapshot**.
3. Analyze a different CSV in the existing analysis workflow, then select **Load
   main analysis as Current**. Alternatively upload previously analyzed UTF-8 CSVs
   directly into either monitoring slot and click **Use uploaded reference/current**.
4. Inspect the report, temporal views, emotion details, and current-row drilldowns.
5. Download **Monitoring Summary CSV**, **Emotion Drift CSV**, and **Monitoring
   Signals CSV**. These contain numerical comparisons and deterministic signals.

For a reproducible demonstration, use `data/synthetic_monitoring_current.csv` as
the second raw dataset. These twelve new examples are entirely synthetic and
contain **no model predictions**. Analyze them with the existing classifier before
loading them into monitoring. No fabricated model scores are bundled.

Uploads require `feedback`, `emotion_label`, and `emotion_confidence`. Labels are
trimmed/lowercased and must belong to the existing seven model emotions. Confidence
must be finite and between 0 and 1. Valid prediction columns are reused without
inference. Raw uploads receive instructions to use Analyze Feedback. Existing CSV
size/row/column limits, malformed-record checks, and duplicate-header validation
apply. Input frames are never modified. Rows with both prediction fields empty
remain in input/missingness comparisons; at least one analyzed row is required.
Predicted rows require usable feedback. Exported collision-suffixed prediction
columns must be renamed to canonical names for uploaded monitoring CSVs; loading
main results resolves field mappings automatically.

The report includes:

- **Prediction drift:** aligned counts and shares in the standard seven-emotion
  order, percentage-point changes, and base-2 **Jensen–Shannon divergence**.
  JSD ranges from 0 (identical distributions) to 1 (disjoint distributions).
  This is divergence, not the square-root Jensen–Shannon distance.
- **Confidence drift:** mean, median, quartiles, distributions, and low-confidence
  counts/rates. Both datasets use one configurable threshold, initialized from
  main analysis or the shared default. Uploaded low-confidence booleans are
  recomputed. The empirical **Kolmogorov–Smirnov statistic** is the maximum CDF
  difference (0–1), implemented with NumPy; no p-values are calculated.
- **Data drift:** cleaned-text character/word means, medians, quartiles, and
  distributions; shared course/subject/semester distributions; valid numeric
  rating summaries/distributions; shared-column missingness rates. Categories
  present in only one dataset remain visible with zero counts in the other.
  Context percentages exclude missing values, while missingness uses all rows.
  Invalid dates/ratings are excluded from numerical views, not treated as missing.
- **Ambiguity:** the Phase 5 top-two margin definition (`< 0.10`) is applied only
  when valid full class scores or confidence margins are available in both inputs.
  Invalid/unavailable evidence produces an unavailable message, never a proxy.
  Emotion-specific low-confidence/ambiguity rates require three predictions;
  counts and support remain visible below that minimum.
- **Temporal monitoring:** day/week/month volume, confidence, low-confidence
  share, and emotion composition, using the existing temporal composition utility.
  ISO dates are parsed in UTC; invalid dates are excluded and counted.

Rules live in `MONITORING_THRESHOLDS` in `src/config.py`. Default JSD bands are
minimal below 0.05, noticeable from 0.05, and substantial from 0.15. Review signals
cover JSD ≥ 0.05, mean-confidence drop ≥ 0.05, low-confidence/missingness increase
≥ 10 percentage points, absolute emotion-share change ≥ 10 percentage points, and
absolute relative median word-length change ≥ 25%. Signals show observed values,
thresholds, and units. Insights are deterministic descriptions of actual metrics;
no LLM or causal explanations are used.

**Drift severity thresholds are heuristic monitoring aids, not universal
statistical guarantees.** Drift does not prove accuracy degradation. Prediction
changes may reflect genuine changes in student feedback. Small datasets produce
unstable drift estimates. Confidence is not calibrated accuracy. Longer feedback
and individual emotions are not ranked as better or worse. Meaningful performance
monitoring still requires labeled data and Phase 5 evaluation.

Reference/current frames, source metadata, results, and inspection selections are
isolated under the `monitoring` session state. Snapshots are independent copies and
survive navigation and replacement of main analysis. Replacing one monitoring
input invalidates its comparison and filters while preserving the other snapshot;
invalid uploads preserve the previous valid selection and display an error.
Changing the confidence threshold invalidates monitoring calculations only.
Results are reused across unrelated reruns. **Snapshots are session-based** and
are not durable history across app restarts. No database or background service
has been added.
