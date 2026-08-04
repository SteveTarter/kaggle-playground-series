# AGENTS.md — PS-S06E08: Predicting Smartphone Addiction

**Competition:** Kaggle Playground Series Season 6, Episode 8  
**Task:** Binary Classification — `addicted_label` (0 = Not Addicted, 1 = Addicted)  
**Metric:** ROC AUC (Area Under the Receiver Operating Characteristic Curve)  
**Branch:** `s06e08-predicting-smartphone-addiction`  
**Environment:** Kaggle Notebooks, Python 3.10, GPU T4 x2  

---

## Role

You are a data science agent working on a Kaggle tabular ML competition. Your scope
is limited to files in this directory (`2026/s06e08/`). Do not modify root-level
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
### Shared Foundation
`ps_s06e08_experiment_setup`  
All notebooks import `ExperimentSetup` to handle parameter management, artifact saves/loads, and dataset uploads.

```python
from ps_s06e08_experiment_setup import ExperimentSetup
setup = ExperimentSetup(
    model_name='LightGBM',
    use_gpu=True,
    perform_rfe=True,
    perform_optuna_tuning=True
)
training_df = setup.read_dataset('training')
```
#### Artifacts Management
Artifacts (selected feature lists and optimal Optuna parameters) are cached locally and synced to a Kaggle dataset:

- **Local Location**: Save `selected_features_<model>.json` and `optuna_params_<model>.json` directly in 2026/s06e08/.
- **Kaggle Location**: Saved to the dataset `stephentarter/ps-s06e08-artifacts`.
- **Kaggle Sync**: Use `setup.upload_artifact(filepath)` to upload newly updated local parameters/RFE files to Kaggle.

### Fold Cache Pattern
To prevent re-running feature engineering on every Optuna trial:

```python
# Build once, cache per fold
fold_cache = {}
for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X, y)):
    fold_cache[fold_idx] = (X[tr_idx], X[val_idx], y[tr_idx], y[val_idx])
# Inside objective(): load from cache, never re-engineer features
X_tr, X_val, y_tr, y_val = fold_cache[fold_idx]
```

### Multi-Seed Inference
Models must evaluate and predict using average blending over multiple seeds (e.g. `[10301, 42, 2026, 777, 888]`) to stabilize scores and prevent overfitting to a single validation split.

## Boundaries

### Always Do
- Run the full notebook from top to bottom before completing a task.
- Retain the exact column output for `save_probabilities()` to ensure OOF and test probabilities align perfectly.
- Use the fold cache pattern in any function called by Optuna.
- Apply Stratified K-Fold cross-validation due to class imbalance.

### Ask First
- Before changing the list of standard seeds or the number of CV splits.
- Before introducing new external datasets.

### Never Do
- Never commit `submission.csv` or executed notebook output files (`*_out.ipynb`).
- Never push secrets, Kaggle API tokens, or credentials.
