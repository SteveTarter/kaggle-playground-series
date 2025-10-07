# Repository Guidelines

## Project Structure & Module Organization
The repository groups Kaggle Playground entries by year under directories like `2025/`, then by episode (`s5e8`, `s5e9`, `s5e10`). Each episode folder contains one or more notebooks, raw CSVs inside `original_data/`, working feature sets in `data/`, and the latest `submission.csv`. Keep exploratory notebooks in the episode folder; promote reusable helpers into lightweight modules only when they outgrow a few cells.

## Build, Test, and Development Commands
Create a local environment that mirrors Kaggle: `python -m venv .venv && source .venv/bin/activate` followed by `pip install jupyter pandas numpy scikit-learn lightgbm xgboost`. Launch notebooks with `jupyter lab 2025/s5e10` or use `kaggle kernels push` when iterating in the hosted environment. Rebuild submissions by running the notebook top-to-bottom and persist generated artifacts in the episode’s `data/` directory or as `submission.csv`.

## Coding Style & Naming Conventions
Target Python 3.10+, 4-space indentation, and `snake_case` for variables, helper functions, and utility modules. Follow Camel Case titles without underscores for notebook filenames (e.g., `Road Accident Risk Prediction (S5E10).ipynb`). Keep cells focused; lift repeated logic into helper functions declared near the top. Use Markdown cells to record data sources, feature engineering notes, and leaderboard scores.

## Testing Guidelines
Validation lives inside the notebooks. Add lightweight assertion cells (for example, `assert len(train_df) == 9557`) after major preprocessing steps. Track performance with stratified KFold or repeated train/validation splits and log metrics to a summary cell. Before replacing `submission.csv`, compare new metrics with prior runs and keep notable alternates under timestamped filenames inside `data/`.

## Commit & Pull Request Guidelines
History favors concise, imperative commit subjects such as “Add K-Fold training.” Keep commits scoped to one modeling or data change and avoid checking in large raw downloads. Pull requests should explain modeling intent, link any related Kaggle discussion or issue, and attach validation outputs or screenshots. Call out new dependencies or environment tweaks so teammates can reproduce results quickly.

## Submission Workflow
Store Kaggle-ready uploads as `submission.csv` within each episode folder and tag the producing notebook cell. Note the public leaderboard score in a Markdown cell so future contributors can trace performance changes.
