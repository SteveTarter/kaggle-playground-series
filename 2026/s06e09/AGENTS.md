# AGENTS.md — PS-S06E09: Electric Vehicle Purchases

**Competition:** Kaggle Playground Series Season 6, Episode 9  
**Task:** Predict a probability for the Will_Buy_EV variable  
**Metric:** ROC AUC  
**Branch:** `s06e09-electric-vehicle-purchases`  
**Environment:** Kaggle Notebooks, Python 3.10, GPU T4 x2  

---

## Role

You are a data science agent working on Kaggle Playground Series S06E09. Your scope is strictly limited to files in `2026/s06e09/`. Do not edit root files or other competition directories.

---

## Commands

Notebooks run on Kaggle. To simulate locally:

```bash
# Install dependencies (match Kaggle's pinned versions)
pip install xgboost==2.1 lightgbm==4.3 catboost==1.2 optuna==3.6 \
            scikit-learn==1.4 torch==2.2 pandas numpy sweetviz

# Run headless notebook execution for testing
jupyter nbconvert --to notebook --execute <notebook>.ipynb --output <notebook>_out.ipynb

# Analyze Optuna study DB and check parameter boundaries
python3 ../../.agents/skills/optuna-study-analyzer/scripts/analyze_study.py \
  --db 2026/s06e09/xgb_study.db \
  --output 2026/s06e09/optuna_params_xgb.json
---

## Architecture

### Shared Foundation:
#### `ps_s06e09_experiment_setup`
All notebooks import `ExperimentSetup`. It is used as a container for configuration settings,
reading datasets, and other utility functions.

```python
from ps_s06e09_experiment_setup import ExperimentSetup

setup = ExperimentSetup(
    model_name='CatBoost',
    use_gpu=True,
    perform_rfe=True,
    perform_optuna_tuning=True
)

training_df = setup.read_dataset('training')
```

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

- Use the fold cache pattern in any function called by Optuna.
- Verify that the accompanying markdown documentation reflects the latest implementation.

### Ask First

- Before changing `n_splits`, `random_state`, or the stratification approach.
- Before blending weights in ensembles.
- Before upgrading any library version.

### Never Do

- Never commit `submission.csv` or `*_out.ipynb` (executed notebook artifacts).
- Never push secrets, Kaggle API tokens, or credentials.


