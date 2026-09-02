import os
import sys
import math
import random
import warnings
from typing import Iterable
from IPython.display import display, Markdown, IFrame

import numpy as np
import pandas as pd
from IPython.display import display
import scipy.stats as st
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, target=None, seed=10301):
        self.target = target
        self.seed = seed
        
    def split_columns(self, df: pd.DataFrame, max_cardinality: int = 30):
        """
        Heuristic split into numeric vs categorical, with a cardinality cap for cats.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        
        # treat low-unique integer columns as categorical (IDs will be filtered later)
        for c in list(num_cols):
            if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique(dropna=True) <= max_cardinality:
                cat_cols.append(c)
                num_cols.remove(c)
                
        # drop high-cardinality cats from categorical plotting by default
        low_card_cats = [c for c in cat_cols if df[c].nunique(dropna=True) <= max_cardinality]
        
        return num_cols, low_card_cats
        
    def collapse_rare_categories(self, s: pd.Series, min_count: int = 50) -> pd.Series:
        vc = s.value_counts()
        keep = vc[vc >= min_count].index
        return s.where(s.isin(keep), other='__OTHER__')

    def categorical_vs_target_classification(self, df: pd.DataFrame, feature, min_count = 50, top_n = 20):
        s = self.collapse_rare_categories(df[feature].astype('object'), min_count=min_count)
        d = pd.DataFrame({feature: s, self.target: df[self.target]}).dropna()
        grp = d.groupby(feature)
        counts = grp.size().sort_values(ascending=False).head(top_n)
        cats = counts.index
    
        rate = grp[self.target].mean().reindex(cats)
    
        # counts
        plt.figure(figsize=(8, max(2, 0.35 * len(cats))))
        plt.barh([str(c) for c in cats], counts.values)
        plt.gca().invert_yaxis()
        plt.xlabel('Count'); plt.title(f'{feature} – Top {len(cats)} Counts')
        plt.show()
    
        # target rates
        plt.figure(figsize=(8, max(2, 0.35 * len(cats))))
        plt.barh([str(c) for c in cats], rate.values)
        plt.gca().invert_yaxis()
        plt.xlabel(f'Mean {self.target}'); plt.title(f'{feature} – Target Rate (Top {len(cats)})')
        plt.show()

    def has_target(self, df: pd.DataFrame):
        return self.target is not None and self.target in df.columns

    def numeric_by_category_trend(self, df: pd.DataFrame, xcol:str, cat:str, q:int = 15, min_count:int = 2000):
        if not self.has_target(df):
            print(f"Dataframe doesn't have a target column {self.target}.")
            return
            
        # Keep frequent categories to reduce noise
        keep = df[cat].value_counts()
        keep = keep[keep >= min_count].index
        for k in keep:
            d = df[df[cat] == k][[xcol, self.target]].dropna()
            if d.empty or d[xcol].nunique() < 2 : 
                continue
                
            bins = pd.qcut(d[xcol], q = min(q, d[xcol].nunique()), duplicates = 'drop')
            m = d.groupby(bins)[self.target].mean()
    
            idx = m.index
            try:
                centers = idx.mid.to_numpy()
            except AttributeError:
                # fallback for very old pandas
                centers = np.array([(iv.left + iv.right) * 0.5 for iv in idx])
                
            plt.plot(centers, m.values, label = str(k))
            plt.title(f'{xcol} → mean({self.target}) by {cat}')
            plt.xlabel(xcol); 
            plt.ylabel(f'Mean {self.target}')
            plt.legend(loc='best')
            
        plt.show()

    def cat_cat_heatmap(self, df: pd.DataFrame, cat1:str, cat2:str, min_count:int = 500):
        if not self.has_target(df):
            print(f"Dataframe doesn't have a target column {self.target}.")
            return
            
        d = df[[cat1, cat2, self.target]].dropna()
        g = d.groupby([cat1, cat2])[self.target].agg(['mean', 'count']).reset_index()
        g = g[g['count'] >= min_count]
        if g.empty:
            print(f'[info] No {cat1}×{cat2} cells with count >= {min_count}.')
            return
    
        # Normalize labels ONCE to consistent strings
        s1 = g[cat1].astype('string').fillna('<NA>').astype(str)
        s2 = g[cat2].astype('string').fillna('<NA>').astype(str)
    
        # Stable ordering (by label) or by frequency if preferred
        rows, r_codes = np.unique(s1, return_inverse = True)
        cols, c_codes = np.unique(s2, return_inverse = True)
    
        A = np.full((len(rows), len(cols)), np.nan, dtype = float)
        A[r_codes, c_codes] = g['mean'].to_numpy()
    
        plt.figure(figsize=(1.2 * len(cols) + 2, 1.2 * len(rows) + 2))
        im = plt.imshow(A, aspect = 'auto', origin = 'upper')
        cbar = plt.colorbar(im, fraction = 0.046, pad = 0.04, label = f'Mean {self.target}')
        plt.xticks(np.arange(len(cols)), cols, rotation = 45, ha = 'right')
        plt.yticks(np.arange(len(rows)), rows)
        plt.title(f'{cat1} × {cat2} → mean({self.target})')
        plt.tight_layout()
        plt.show()

    def numeric_numeric_hex(self, df: pd.DataFrame, x:str, y:str, gridsize:int = 50):
        if not self.has_target(df):
            print(f"Dataframe doesn't have a target column {self.target}.")
            return
            
        d = df[[x, y, self.target]].dropna()
        plt.figure(figsize = (7, 5))
        hb = plt.hexbin(d[x].to_numpy(), d[y].to_numpy(), C=d[self.target].to_numpy(),
                        gridsize = gridsize, reduce_C_function = np.mean)
        plt.xlabel(x); plt.ylabel(y); plt.title(f'{x} × {y} → mean({self.target})')
        cb = plt.colorbar(hb)
        cb.set_label(f'Mean {self.target}')
        plt.show()

    # Feature signal ranking without modeling
    def show_feature_signal_ranking(self, df: pd.DataFrame):
        if not self.has_target(df):
            print(f"Dataframe doesn't have a target column {self.target}.")
            return
            
        # Spearman for numeric; ANOVA-style effect for categoricals
        num_cols = df.select_dtypes(include = [np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != self.target]
    
        scores = []
        for c in num_cols:
            s = df[[c, self.target]].dropna()
            if s.empty: continue
            rho, p = st.spearmanr(s[c], s[self.target])
            scores.append((c, float(rho)))
        scores = sorted(scores, key = lambda x: -abs(x[1]))
        print('Top numeric (|Spearman|):', scores[:10])
    
        cat_cols = [c for c in df.columns if c not in num_cols + [self.target]]
        effects = []
        for c in cat_cols:
            g = df.groupby(c, observed = True)[self.target].mean()
            if g.size >= 2:
                rng = float(g.max() - g.min())
                effects.append((c, rng, int(df[c].nunique())))
                
        effects = sorted(effects, key = lambda x: -x[1])
        print('Top categorical (range of mean target):', effects[:10])

    def plot_pairplot(self, df: pd.DataFrame, num_features: Iterable[str]):
        """
        Generates a pair plot for numerical features, colored by the target variable.
        """
        sns.pairplot(df[num_features + [self.target]], hue=self.target, palette = 'viridis', height = 6)
    
        plt.suptitle('Pair Plot of Numerical Features', y=1.02)
        plt.show()
        
    def plot_boxplots_grid(self, df: pd.DataFrame, cat_features: Iterable[str], n_cols: int = 3):
        """
        Generates box plots for each categorical feature against the target variable
        in a n_cols-column grid.
        """
        n_features = len(cat_features)
        n_rows = math.ceil(n_features / n_cols)
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize = (18, n_rows * 5))
        axes = axes.flatten()
    
        for i, feature in enumerate(cat_features):
            ax = axes[i]
            sns.boxplot(x=feature, y=self.target, data=df, palette='viridis', ax = ax)
            ax.set_title(f'Box Plot of {self.target} by {feature}', fontsize = 12)
            ax.tick_params(axis = 'x', labelrotation = 45)
    
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
    
        plt.tight_layout()
        plt.show()
    
    def plot_violinplots_grid(self, df: pd.DataFrame, cat_features: Iterable[str], n_cols: int = 3):
        """
        Generates violin plots for each categorical feature against the target variable.
        in a n_cols-column grid.
        """
        n_features = len(cat_features)
        n_rows = math.ceil(n_features / n_cols)
 
        if not n_rows:
            print('No categorical features found. Skipping violin plots')
            return
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize = (18, n_rows * 5))
        axes = axes.flatten()
    
        for i, feature in enumerate(cat_features):
            ax = axes[i]
            sns.violinplot(x=feature, y=self.target, data = df, palette = 'viridis', ax = ax)
            ax.set_title(f'Violin Plot of {self.target} by {feature}', fontsize = 12)
            ax.tick_params(axis = 'x', labelrotation = 45)
    
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
    
        plt.tight_layout()
        plt.show()        

    def _coerce_numeric_series(self, s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s
        return pd.to_numeric(
            s.astype(str)
             .str.replace(r'[,\\$%]', '', regex=True)
             .str.replace(r'\\s+', '', regex=True)
             .replace({'n/a': np.nan, 'na': np.nan, '-': np.nan, '': np.nan}),
            errors='coerce'
        )
    
    def numeric_vs_target_classification(self, df: pd.DataFrame, numeric_col: str):
        """
        Plots the distribution of a single numeric feature against a categorical target
        using a boxplot and a violin plot.
        """
        if numeric_col not in df.columns:
            print(f'[warn] Numeric column "{numeric_col}" not in DataFrame. Skipping plot.')
            return
        if self.target not in df.columns:
            print(f'[warn] Target column "{self.target}" not in DataFrame. Skipping plot.')
            return
    
        plot_df = df[[numeric_col, self.target]].copy()
        plot_df[numeric_col] = self._coerce_numeric_series(plot_df[numeric_col])
        plot_df[self.target] = plot_df[self.target].astype('category')
        plot_df = plot_df.dropna(subset=[numeric_col, self.target])
    
        if plot_df.empty:
            print(f'[warn] No valid data to plot for {numeric_col} vs {self.target}. Skipping.')
            return
    
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Boxplot
            sns.boxplot(x=self.target, y=numeric_col, data=plot_df, ax=axes[0], palette='viridis')
            axes[0].set_title(f'Boxplot of {numeric_col}')
            axes[0].set_xlabel(self.target)
            axes[0].set_ylabel(numeric_col)
            
            # Violin Plot
            sns.violinplot(x=self.target, y=numeric_col, data=plot_df, ax=axes[1], palette='viridis')
            axes[1].set_title(f'Violin Plot of {numeric_col}')
            axes[1].set_xlabel(self.target)
            axes[1].set_ylabel(numeric_col)
            
            fig.suptitle(f'{numeric_col} vs. {self.target} Distribution', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f'[warn] Plotting failed for {numeric_col} vs {self.target}: {e}')

    def top_numeric_by_spearman(self, df: pd.DataFrame, num_cols: Iterable[str], k = 6):
        import scipy.stats as st
        scores = []
        for c in num_cols:
            d = df[[c, self.target]].dropna()
            if d.empty: 
                continue
                
            # skip all-constant / all-equal
            if d[c].nunique() < 2: 
                continue
                
            rho, _ = st.spearmanr(d[c], d[self.target])
            if np.isfinite(rho):
                scores.append((c, abs(float(rho))))
                
        scores.sort(key=lambda x: -x[1])
        return [c for c,_ in scores[:k]]
    
    def top_categorical_by_range(self, df: pd.DataFrame, cat_cols: Iterable[str], min_count:int = 200, k:int = 6):
        effects = []
        for c in cat_cols:
            d = df[[c, self.target]].dropna()
            if d.empty: 
                continue
                
            vc = d[c].value_counts()
            keep = set(vc[vc >= min_count].index)
            if not keep:
                continue
                
            m = d[d[c].isin(keep)].groupby(c)[self.target].mean()
            if m.size >= 2:
                effects.append((c, float(m.max() - m.min())))
                
        effects.sort(key=lambda x: -x[1])
        return [c for c,_ in effects[:k]]

    def run_data_visualization(self, df: pd.DataFrame, max_cat_card: int = 30,
                 heavy_sample: int = 150_000):
        """
        Compact data_visualization runner for mixed tabular data.
        - Uses target-aware visuals when target is present.
        - Skips safely on test sets (no target).
        - Limits heavy plots to top-signal features.
        - Optionally downsamples for hexbin/heatmaps.
        """
        df_wo_target = df.drop(columns = [self.target], errors = 'ignore') if self.target else df
    
        num_cols, cat_cols = self.split_columns(df_wo_target, max_cardinality = max_cat_card)
        num_count = len(num_cols)
        cat_count = len(cat_cols)
        
        bool_cols = [c for c in df_wo_target.columns if pd.api.types.is_bool_dtype(df_wo_target[c])]
        cat_cols = sorted(set(cat_cols).union(bool_cols))
    
        if not self.has_target(df):
            print('Target not present → skipping target-aware plots.')
            return
    
        y = df[self.target]
        is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique(dropna = True) > 20
    
        if is_regression:
            top_nums = self.top_numeric_by_spearman(df, [c for c in num_cols if c != self.target], k = num_count) or num_cols[:num_count]
            top_cats = self.top_categorical_by_range(df, cat_cols, min_count = 200, k = cat_count) or cat_cols[:cat_count]
    
            display(Markdown('### Pair Plots'))
            self.plot_pairplot(df, top_nums)
    
            display(Markdown('### Violin Plots'))
            self.plot_violinplots_grid(df, top_cats)
    
            if top_nums and top_cats:
                display(Markdown('### Numeric × Categorical (Trend By Category)'))
                self.numeric_by_category_trend(df, top_nums[0], top_cats[0], q = 15, min_count = 3000)
    
            if len(top_cats) >= 2:
                display(Markdown('### Categorical × Categorical (Heatmap)'))
                self.cat_cat_heatmap(df, top_cats[0], top_cats[1], min_count = 1000)
    
            if len(top_nums) >= 2:
                display(Markdown('### Numeric × Numeric (Hexbin Target Mean)'))
                d_hex = df
                if heavy_sample is not None and len(df) > heavy_sample:
                    d_hex = df.sample(heavy_sample, random_state = self.seed)
                self.numeric_numeric_hex(d_hex, top_nums[0], top_nums[1], gridsize = 50)
    
        else:
            top_nums = [c for c in num_cols if c != self.target][:num_count]
            top_cats = cat_cols[:cat_count]
    
            for c in top_nums:
                try:
                    self.numeric_vs_target_classification(df, c)
                except Exception as e:
                    print(f'[warn] numeric(classif) plot failed for {c}: {e}')
    
            for c in top_cats:
                try:
                    self.categorical_vs_target_classification(df, c, min_count = 100, top_n = 20)
                except Exception as e:
                    print(f'[warn] categorical(classif) plot failed for {c}: {e}')
    
            if len(top_nums) and len(top_cats):
                display(Markdown('### Numeric x Category Trend'))
                self.numeric_by_category_trend(df, top_nums[0], top_cats[0], q = 12, min_count = 3000)
            if len(top_cats) >= 2:
                display(Markdown('### Category x Category Trend'))
                self.cat_cat_heatmap(df, top_cats[0], top_cats[1], min_count = 1500)
    
        display(Markdown('### Feature Signal Ranking'))
        self.show_feature_signal_ranking(df)
