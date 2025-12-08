import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

class ModelVisualizer:
    """
    Centralizes all visualization logic for Gradient Boosting models (XGBoost/CatBoost).
    """
    def __init__(self, palette='viridis', model_name=None):
        self.palette = palette
        self.model_name=model_name
        plt.style.use('seaborn-v0_8-whitegrid') # Optional: Set a nice style

    def plot_learning_curves(self, eval_results, metric='auc'):
        """
        Plots Train vs Val metrics over iterations for multiple folds.
        Expects eval_results from XGBoost/LightGBM (validation_0, validation_1)
        or CatBoost (learn, validation_0).
        """
        plt.figure(figsize=(10, 6))
        
        for i, result in enumerate(eval_results):
            # Detect Keys dynamically based on Model Type
            keys = list(result.keys())
            
            # Defaults
            train_key, val_key = None, None

            if self.model_name == 'CatBoost':
                train_key = 'learn'
                # CatBoost val key often 'validation_0' or 'validation'
                val_key = 'validation_0' if 'validation_0' in keys else keys[-1]
                
            elif self.model_name == 'LightGBM':
                # LightGBM usually uses 'training' and 'valid_1'
                train_key = 'training' if 'training' in keys else keys[0]
                val_key = 'valid_1' if 'valid_1' in keys else keys[-1]
                
            else: # XGBoost default
                train_key = 'validation_0'
                val_key = 'validation_1'

            # Safety Check
            if train_key not in result or val_key not in result:
                print(f"Warning: Could not find keys '{train_key}' or '{val_key}' in fold {i+1}. Found: {keys}")
                continue

            # Extract Metrics
            # LightGBM/XGBoost dicts often have the metric name nested, e.g. {'auc': [...]}
            # CatBoost often has it directly or nested depending on API version.
            try:
                # Handle nested dictionary case (Standard XGB/LGBM)
                if isinstance(result[train_key], dict):
                    train_metric = result[train_key][metric]
                    val_metric = result[val_key][metric]
                else:
                    # Handle flat case (Some CatBoost versions)
                    train_metric = result[train_key]
                    val_metric = result[val_key]
            except KeyError:
                print(f"Error: Metric '{metric}' not found in keys. Available metrics: {result[train_key].keys()}")
                return

            # Plot
            lbl_train = 'Train' if i == 0 else None
            lbl_val = 'Val' if i == 0 else None
            
            plt.plot(train_metric, color='blue', alpha=0.3, label=lbl_train)
            plt.plot(val_metric, color='red', alpha=0.6, linewidth=1.5, label=lbl_val)

        plt.title(f'{self.model_name} Learning Curves ({metric.upper()})')
        plt.xlabel('Iterations')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    def plot_feature_importance(self, models, top_n=30, show_values=True):
        """
        Aggregates and plots feature importance across multiple trained models.
        """
        if not models:
            print("No models provided for feature importance.")
            return

        feature_importance = pd.DataFrame()
        
        for i, model in enumerate(models):
            # XGBoost specific: get_score(importance_type='gain')
            # If using CatBoost, change to: model.get_feature_importance()
            try:
                imp_dict = model.get_booster().get_score(importance_type='gain')
            except AttributeError:
                # Fallback for Sklearn API wrapper consistency or other models
                if hasattr(model, 'feature_importances_'):
                    imp_dict = dict(zip(model.feature_names_in_, model.feature_importances_))
                else:
                    continue

            fold_imp = pd.DataFrame({
                'Feature': list(imp_dict.keys()),
                'Importance': list(imp_dict.values()),
                'Fold': i + 1
            })
            feature_importance = pd.concat([feature_importance, fold_imp], axis=0)

        # Aggregation
        avg_imp = feature_importance.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(10, 12))
        ax = sns.barplot(x=avg_imp.values, y=avg_imp.index, palette=self.palette)

        if show_values:
            for i in range(top_n):
                ax.bar_label(ax.containers[i])

        plt.title(f'{self.model_name} Top {top_n} Feature Importances (Avg Gain)')
        plt.xlabel('Gain')
        plt.tight_layout()
        plt.show()

    
    def plot_roc_curve(self, y_true, y_preds):
        """
        Plots the ROC curve for Out-of-Fold predictions.
        """
        auc_score = roc_auc_score(y_true, y_preds)
        fpr, tpr, _ = roc_curve(y_true, y_preds)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()