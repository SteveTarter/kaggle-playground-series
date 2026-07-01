# AGENTS.md — PS-S06E07: Predicting Student Health Risk

**Competition:** Kaggle Playground Series Season 6, Episode 7
**Task:** Three-class classification — `at-risk`, `unhealthy`, `fit`
**Metric:** Balanced Accuracy (macro-averaged recall)
**Branch:** `s06e07-predicting-student-health-risk`
**Environment:** Kaggle Notebooks, Python 3.10, GPU T4 x2

---

## Role

You are a data science agent working on a Kaggle tabular ML competition. Your scope
is limited to files in this directory (`2026/s06e07/`). Do not modify root-level
files, CI config, or other competition directories.

---

## Commands

Notebooks run on Kaggle. To simulate locally:

```bash
# Install dependencies (match Kaggle's pinned versions)
pip install xgboost==2.1 lightgbm==4.3 catboost==1.2 optuna==3.6 \
            scikit-learn==1.4 torch==2.2 pandas numpy sweetviz
```

---

## Architecture

### Shared Foundation:
#### `ps_s06e07_experiment_setup`
All notebooks import `ExperimentSetup`. It is used as a container for configuration settings,
reading datasets, and other utility functions.

```python
from ps_s06e07_experiment_setup import ExperimentSetup

setup = ExperimentSetup(
    model_name='CatBoost',
    use_gpu=True,
    perform_rfe=True,
    perform_optuna_tuning=True
)

training_df = setup.read_dataset('training')
```

### Target Mapping

This mapping is **standardized across all notebooks** and must not drift:

```python
TARGET_MAPPING = {"at-risk": 0, "unhealthy": 1, "fit": 2}
INVERSE_TARGET_MAPPING = {0: "at-risk", 1: "unhealthy", 2: "fit"}
```

Use `INVERSE_TARGET_MAPPING` when decoding predictions for `submission.csv`.

### Fold Cache Pattern

To avoid re-running feature engineering on every Optuna trial:

```python
# Build once, cache per fold
fold_cache = {}
for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X, y)):
    fold_cache[fold_idx] = (X[tr_idx], X[val_idx], y[tr_idx], y[val_idx])

# Inside objective(): load from cache, never re-engineer features
X_tr, X_val, y_tr, y_val = fold_cache[fold_idx]
```

---

## Boundaries

### Always Do

- Run the full notebook from top to bottom before submitting any changes.
- Keep `TARGET_MAPPING` identical across all notebooks.
- Use the fold cache pattern in any function called by Optuna.

### Ask First

- Before changing `n_splits`, `random_state`, or the stratification approach.
- Before blending weights in ensembles.
- Before upgrading any library version.

### Never Do

- Never use a different `TARGET_MAPPING` in a new notebook.
- Never commit `submission.csv` or `*_out.ipynb` (executed notebook artifacts).
- Never push secrets, Kaggle API tokens, or credentials.
