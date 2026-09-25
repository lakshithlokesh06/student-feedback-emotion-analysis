# Phase 7 validation — 2026-09-25

- Default suite: 201 passed, 2 opt-in integration tests skipped.
- Full suite with cached real model: `RUN_MODEL_INTEGRATION=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q` — **203 passed**, no skips.
- `python -m compileall -q app.py src tests` — passed.
- Imported every module under `src` — passed.
- `.venv/bin/python -m pip check` — no broken requirements; no dependencies added.
- `git diff --check` — passed.
- Streamlit startup on localhost:8501 — passed after allowing local port binding.

## Manual browser smoke test

1. Classified all 48 bundled student-feedback examples using the real unchanged model.
2. Captured the main analysis as a reference snapshot.
3. Uploaded and classified the 12 new raw synthetic monitoring examples with the same model.
4. Loaded main analysis as current; preserved the 48-row baseline.
5. Verified dataset summaries: reference neutral, 77.9% mean confidence, 14.6% low confidence; current fear, 92.9% mean confidence, 0% low confidence.
6. Verified emotion counts/shares and grouped bars: fear 4/48 versus 7/12, a +50 percentage-point change. JSD 0.268.
7. Verified confidence distributions, statistics, and KS statistic 0.583.
8. Verified word/character summaries and distributions: median words 12.5 versus 9.5.
9. Verified available context controls, expanded course comparison (including current-only New Course), rating metrics, and missingness comparisons.
10. Expanded temporal monitoring and verified volume, confidence, low-confidence, and emotion composition charts.
11. Verified deterministic review signals and descriptive summary; ambiguity correctly unavailable for ordinary analysis without full scores/margins.
12. Expanded emotion inspection and current-row drilldown; empty low-confidence selection handled correctly.
13. Triggered each of the three monitoring CSV downloads and received all three browser download events. Export schemas/content are covered by unit tests.
14. Navigated to Model Evaluation and completed real inference on all 49 labeled sample rows; performance tables and confusion matrix rendered.
15. Navigated to Human Review and verified the 12-row main analysis queue, prediction details, and annotation controls.
16. Returned to monitoring and verified both 48-row reference and 12-row current snapshots persisted.

The browser workflow exercised main-analysis snapshots. Analyzed CSV validation,
malformed inputs, missing optional data, ambiguity evidence, state invalidation,
threshold changes, and export content were checked by automated tests. No model
outputs were fabricated. No deployment, Git commit, or Git push was performed.
