# Known issues for follow-up PRs

These problems are outside the chat and trip-planner fix. Each section is meant to become its own focused PR. The ML pipeline doesn't feed the live web app, so none of these affect what riders see. They do affect what the project claims about its model.

Line numbers refer to `main` at f4bc559.

## 1. The delay label measures dwell time, not lateness

- `Model_Development/ml_src/model_train.py:40-44` defines `delay_minutes = departure_time - arrival_time` and `delayed = delay_minutes > 5`. Tuning, selection, fairness and explainability all reuse this label through `prepare_data`.
- The inputs are MBTA `/predictions` attributes only (`Data_Pipeline/scripts/preprocess_data.py:58-65`). No schedule or actual-arrival data is collected, so the label is the predicted time a vehicle spends at a stop.
- In the committed `Model_Development/models/reference_stats.json`, **5 of 127,159 rows** are labelled delayed.
- `Model_Development/ml_src/preprocess_events.py:83-86` computes a real lateness label (`actual - scheduled > 2`), but nothing calls it. Its input folder doesn't exist, and its threshold (2 min) disagrees with training (5 min).

**Fix:** compute `actual − scheduled`, either by joining predictions to schedules or from the events data. Use one configured threshold, and check the class balance before training.

## 2. The evaluation leaks

- SMOTE runs on the whole dataset before `train_test_split` (`model_tuning.py:57-64`), so synthetic copies of the few positives land in both train and validation.
- The split is random and stratified on time-ordered snapshots (`model_train.py:82-84`). Only exact duplicates are dropped (`preprocess_data.py:70`), so repeated polls of the same trip can fall on both sides.
- `model_select.py:51,82,89` scores every candidate on the full dataset, training rows included, and picks the winner by that AUC.
- As a result, the reported ROC-AUC of 0.81 is in-sample and rests on about 5 positives. F1 is 0.00015.

**Fix:** split by time or by trip, run SMOTE inside the training fold only (an imblearn `Pipeline`), and keep a held-out test set for selection.

## 3. The quality checks are placeholders

- `Data_Pipeline/scripts/run_quality_checks.py:10-16` returns four hardcoded `PASS` rows ("Simulated quality checks").
- **The schema check can't pass and can't stop anything:**
  - `params.yaml` expects columns such as `attributes.arrival_time`, but preprocessing writes flat names, so the check always fails.
  - `validate_schema.py` only writes that status to a file and never raises.
- `detect_anomalies.py:18` looks for `attributes.arrival_time`, so the missing-arrival rule never fires. Alerts get no checks at all.
- `detect_bias.py:47-56` builds random, simulated labels and predictions.
- **Great Expectations and TFDV are imported but not used for validation:**
  - Great Expectations appears only as `ge.from_pandas`, and the result is unused (`validate_schema_profile.py:32`).
  - `validate_schema_tfdv.py` uses neither library.
- Fetch errors are logged and the pipeline carries on (`fetch_data.py:63-66,74-82`).

**Fix:** match the expected columns to the processed output, fail on violations, replace the simulated checks with real ones, and remove the random-label bias step.

## 4. The pipeline can't be reproduced

- `.dvc/config` is empty, so there is no DVC remote, and `dvc pull` can't fetch data. `Data_Pipeline/data/` holds only `.gitignore`.
- `dvc.yaml` stages depend on a root-level `params.yaml`, but the file lives at `Data_Pipeline/params.yaml`. The scripts open it relative to the working directory.
- The DAGs `cd /opt/airflow/Data_Pipeline` and then read `Data_Pipeline/data/raw`, which doubles the path.
- There are no DVC stages for train, tune or select.
- `requirements.txt` lacks lightgbm, mlflow, imbalanced-learn, shap and lime. CI installs them unpinned.
- Several steps log a warning and exit 0 when their input is missing (`model_train.py:65-69,78-80`, `model_tuning.py:34-38,53-55`, `register_model.py:76-77`).

**Fix:** configure a remote, fix the paths and working directories, add the ML stages to DVC, pin dependencies, and exit non-zero on failure.

## 5. The drift numbers are invalid

- `Model_Development/ml_src/monitor_drift.py:27-28` lets `np.histogram` choose separate bin edges for the reference and the current data, which makes PSI invalid.
- The "current" data is the same `predictions.csv` as the reference, minus some row filtering. PSI is therefore near 0 and can never reach the retrain threshold.
- The KS test flags drift at p < 0.05 with about 127k rows, so trivial differences count as drift.
- Thresholds differ between `monitoring/drift_monitor.py` (0.1 / 0.3) and retraining (0.3).
- `reports/drift_report.json` is truncated and isn't valid JSON.
- The README mentions Evidently, but no Evidently code exists.

**Fix:** take bin edges from the reference data's quantiles, compare against a newer data window, and keep one threshold configuration.

## 6. Retraining never promotes, and registration is unconditional

- `retraining/trigger_retrain.py:62-63` reads `baseline_lgbm_auc` and `final_model_existing_auc`. `model_comparison.json` only has `results` and `best_model`, so both values are 0 and the "only if AUC improves" check never passes.
- The same script runs scripts by path with `cwd=ml_src`, which breaks their package imports, and it ignores their exit codes.
- `check_drift` reads a stale report instead of running drift detection.
- `model_select.py:97-98` overwrites `final_model.joblib` before any validation.
- `register_model.py:59-64` promotes to Production unconditionally, and CI runs it.
- `trigger_retrain_with_notify.py` sends hardcoded PSI and AUC values in its notification.
- Paths and names don't line up:
  - CI uploads root `models/` and `mlruns/`, but the scripts write under `Model_Development/`.
  - The bias analysis loads `model_lgbm`, not the final model.

**Fix:** read the real keys, compare the candidate with production on held-out data, gate registration on that comparison, check exit codes, and use one path configuration.

## 7. CI and the docs claim more than the code does

**CI:**
- Every ML step in `.github/workflows/charlie_mbta_ci.yml` is guarded by a file-existence check or `|| echo`.
- On the latest `main` run, every step was skipped or failed for lack of data. The job failed only at the GCP auth step, which needs a `GCP_SA_KEY` secret the repo doesn't have.
- CI never runs `model_fairness`.
- The new `App tests` workflow is separate and doesn't depend on any of this.

**README:**
- The README's MLOps table says data is validated with TFDV and Great Expectations, mentions Evidently drift and an ROC-AUC of 0.81, and says CI trains, tunes, selects and registers. Sections 2–6 show what is actually there.
- The 0.81 figure is cited as `reports/model_comparison.json`; the file is at `Model_Development/reports/`.
- The fairness analysis slices only on `direction_id`, which is itself a model feature, and it has no pass/fail threshold.

**`Model_Development/readme.md`:**
- It calls LightGBM the final winner, but logistic regression is selected.
- It claims 5-fold CV and a confusion matrix, which the code doesn't produce.
- It refers to `src/` and `mlops_pipeline.yml`; the actual names are `ml_src/` and `charlie_mbta_ci.yml`.

**Fix:** remove the `|| echo` masking so CI reports real failures, run GCP steps only where credentials exist, and correct the claims above once sections 1–6 are fixed.

## 8. Other app issues found during the chat fix

- **Leaked key:** `API_Service/chatbot.py:57` hardcodes a default MBTA API key. The file is legacy: the deployed app doesn't import it, and the Dockerfile doesn't copy it. Remove the default and rotate the key, because deleting it doesn't remove it from git history.
- **Dead code:** `chatbot.py`, `API_Service/api/__init__.py` and `patch_destinations.py` appear unused by the deployed app. `deploy.yml`'s "Run tests" job only imports packages.
- **Duplicate listeners:** `static/js/views/overview.js` and `stations.js` attach click listeners to the shared view container on every visit. This is the same bug this PR fixes in the chat view.
- **Swallowed errors:** `get_bus_predictions` and `get_commuter_rail_predictions` in `app.py` swallow request errors and return empty lists. The language model then phrases those empty results.
- **Directory test command:** `node --test tests/js/` fails on Node 24 when given a directory. Use `node --test tests/js/*.test.mjs`.
