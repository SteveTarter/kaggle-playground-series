# AGENTS.md — kaggle-playground-series

This repository contains Kaggle Playground Series competition notebooks, organized
by year and episode (e.g., `2026/s06e06/`). Each competition is self-contained.

## Repo Layout

```
kaggle-playground-series/
├── AGENTS.md                  ← you are here
├── 2026/
│   └── s06e06/
│       ├── AGENTS.md          ← competition-specific instructions (start here)
│       ├── ps_s06e06_feature_engineering.py
│       ├── ps-s06e06-xgboost.ipynb
│       ├── ps-s06e06-lightgbm.ipynb
│       ├── ps-s06e06-catboost.ipynb
│       ├── ps-s06e06-nn-tabular-resnet.ipynb
│       └── ...
└── ...
```

## Ground Rules

- **Each competition directory has its own AGENTS.md.** Always read it before
  touching any files in that directory — it contains the authoritative instructions
  for that competition's conventions, architecture, and boundaries.
- Notebooks are run on Kaggle (Python 3.10, GPU T4 x2). Do not assume a local
  Python environment.
- Never modify `submission.csv` or any output artifact directly; they are produced
  by notebook runs.
- Never add API keys, tokens, or secrets to any file.
- Branches follow the pattern `sSSEEE-short-description` (e.g.,
  `s06e06-predicting-stellar-class`). Work within the active competition branch.

## Adding a New Competition

1. Create `YYYY/sSSeEE/` directory on a new branch.
2. Copy the `FeatureFactory.py` and notebook stubs from the most recent competition
   as a starting point, then adapt.
3. Write an `AGENTS.md` for the new competition before writing any model code.
