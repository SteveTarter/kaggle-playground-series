# AGENTS.md — PS-S06E06: Predicting Stellar Class

**Competition:** Kaggle Playground Series Season 6, Episode 6
**Task:** Three-class classification — `GALAXY`, `QSO`, `STAR`
**Metric:** Balanced Accuracy (macro-averaged recall)
**Branch:** `s06e06-predicting-stellar-class`
**Environment:** Kaggle Notebooks, Python 3.10, GPU T4 x2

---

## Role

You are a data science agent working on a Kaggle tabular ML competition. Your scope
is limited to files in this directory (`2026/s06e06/`). Do not modify root-level
files, CI config, or other competition directories.

---

## Commands

Notebooks run on Kaggle. To simulate locally:

```bash
# Install dependencies (match Kaggle's pinned versions)
pip install xgboost==2.1 lightgbm==4.3 catboost==1.2 optuna==3.6 \
            scikit-learn==1.4 torch==2.2 pandas numpy

# Run a notebook headlessly (for quick smoke-testing)
jupyter nbconvert --to notebook --execute <notebook>.ipynb --output <notebook>_out.ipynb
```

There is no Makefile. All orchestration is done inside individual notebooks.

---

## Architecture

### Shared Foundation: 
#### `ps_s06e06_feature_engineering.py`

All notebooks import `FeatureFactory`. It is the **single source of truth** for
feature engineering. Do not duplicate feature logic inside notebooks.

```python
from ps_s06e06_feature_engineering import FeatureFactory

ff = FeatureFactory(
    strategies=["colors", "ratios", "redshift", "position", "interactions", "encoding"],
    include_flux_features=False   # True for NN only — see below
)
X = ff.fit_transform(df_train)
```
#### `ps_s06e06_experiment_setup`
All notebooks import `ExperimentSetup`. It is used as a container for configuration settings,
reading datasets, and other utility functions.  New, general purpose utility functions should go here.
```python
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
TARGET_MAPPING = {"QSO": 0, "STAR": 1, "GALAXY": 2}
INVERSE_TARGET_MAPPING = {0: "QSO", 1: "STAR", 2: "GALAXY"}
```

Use `INVERSE_TARGET_MAPPING` when decoding predictions for `submission.csv`.

### Fold Cache Pattern

To avoid re-running `FeatureFactory` on every Optuna trial:

```python
# Build once, cache per fold
fold_cache = {}
for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X, y)):
    fold_cache[fold_idx] = (X[tr_idx], X[val_idx], y[tr_idx], y[val_idx])

# Inside objective(): load from cache, never re-engineer features
X_tr, X_val, y_tr, y_val = fold_cache[fold_idx]
```

Never call `FeatureFactory.fit_transform()` inside an Optuna `objective()` function.

---

## Notebook Conventions

### One Model Per Notebook

| Notebook | Model | Notes |
|----------|-------|-------|
| `ps-s06e06-xgboost.ipynb` | XGBoost | device=`cuda`, tree_method=`hist` (not `gpu_hist` — deprecated in XGBoost 2.x) |
| `ps-s06e06-lightgbm.ipynb` | LightGBM | Custom metric; see metric shape note below |
| `ps-s06e06-catboost.ipynb` | CatBoost | Uses `Logloss` loss (not `MultiClass`) — see CatBoost note below |
| `ps-s06e06-nn-tabular-resnet.ipynb` | TabularResNet | `include_flux_features=True`, CrossEntropyLoss with class weights |
| `ps-s06e06-model-blending.ipynb` | Blending | Loads OOF predictions from all model notebooks |
| `ps-s06e06-model-stacking.ipynb` | Stacking | Loads OOF predictions from all model notebooks |

### Optuna Setup (all notebooks)

```python
sampler = optuna.samplers.TPESampler(multivariate=True, seed=setup.get_seed())
study = optuna.create_study(direction="maximize", sampler=sampler)

# Always seed with a known-good trial before searching
study.enqueue_trial({
    "learning_rate": 0.05,
    "max_depth": 6,
    # ... other reasonable defaults
})

study.optimize(objective, n_trials=50, callbacks=[pruning_callback])
```

### K-Fold

Use **`StratifiedKFold(n_splits=5, shuffle=True, random_state=setup.get_seed())`** consistently.
Do not use plain `KFold`; class imbalance makes stratification important.

---

## Known Pitfalls (Read Before Editing)

### XGBoost

- **`gpu_hist` is deprecated.** Use `tree_method="hist"` + `device="cuda"`.
- **`eval_metric` in XGBoost 2.x** must be passed in `train()`, not the constructor:
  ```python
  # WRONG (XGBoost 2.x)
  model = xgb.XGBClassifier(eval_metric="mlogloss")
  # RIGHT
  model.fit(X_tr, y_tr, eval_metric="mlogloss", eval_set=[(X_val, y_val)])
  ```

### LightGBM

- Custom metric functions must return `(name, value, is_higher_better)`.
- The metric receives `(y_pred, dataset)` where `y_pred` is **flattened** for
  multi-class. Reshape before computing balanced accuracy:
  ```python
  def balanced_acc_metric(y_pred, dataset):
      y_true = dataset.get_label().astype(int)
      y_pred_reshaped = y_pred.reshape(n_classes, -1).T  # shape: (n_samples, n_classes)
      y_pred_labels = y_pred_reshaped.argmax(axis=1)
      score = balanced_accuracy_score(y_true, y_pred_labels)
      return "balanced_accuracy", score, True
  ```

### CatBoost

- **`BalancedAccuracy` is incompatible with `MultiClass` loss** in CatBoost.
  Use `loss_function="Logloss"` (one-vs-rest under the hood) instead.
- Do not pass `eval_metric="BalancedAccuracy"` when using `MultiClass`.

### Neural Network

- Uses `CrossEntropyLoss` with **inverse-frequency class weights**:
  ```python
  class_counts = np.bincount(y_train)
  weights = 1.0 / class_counts
  weights = torch.tensor(weights / weights.sum(), dtype=torch.float32).to(device)
  criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
  ```
- Scheduler: `CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)`
- Early stopping has a **minimum epoch threshold** — do not stop before epoch 20
  regardless of validation loss, to avoid stopping in a warm-restart trough.

### Optuna Pruning

- Pruner fires on intermediate values. Report after each fold, not each epoch,
  to avoid over-pruning multi-fold studies:
  ```python
  trial.report(fold_score, step=fold_idx)
  if trial.should_prune():
      raise optuna.exceptions.TrialPruned()
  ```

---

## Boundaries

### Always Do

- Run the full notebook from top to bottom before submitting any changes.
- Keep `TARGET_MAPPING` identical across all notebooks.
- Use the fold cache pattern in any function called by Optuna.
- Add docstrings to new `FeatureFactory` strategy methods.

### Ask First

- Before adding a new feature strategy to `FeatureFactory` — it affects all models.
- Before changing `n_splits`, `random_state`, or the stratification approach.
- Before blending weights in `ensemble.ipynb` — these are tuned empirically.
- Before upgrading any library version — Kaggle pins these.

### Never Do

- Never call `FeatureFactory.fit_transform()` inside an Optuna `objective()`.
- Never use `gpu_hist` as `tree_method` in XGBoost.
- Never use a different `TARGET_MAPPING` in a new notebook.
- Never commit `submission.csv` or `*_out.ipynb` (executed notebook artifacts).
- Never push secrets, Kaggle API tokens, or credentials.
