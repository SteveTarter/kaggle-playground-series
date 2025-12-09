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
        Plots Train vs Val metrics over iterations.
        Handles case sensitivity ('AUC' vs 'auc') and missing training metrics.
        """
        plt.figure(figsize=(10, 6))
        
        for i, result in enumerate(eval_results):
            keys = list(result.keys())
            train_key, val_key = None, None
            
            # Auto-Detect Keys
            if 'learn' in keys: # CatBoost
                train_key = 'learn'
                # Find the validation key (usually validation_0 or validation_1)
                val_keys = [k for k in keys if 'validation' in k]
                val_key = val_keys[-1] if val_keys else None
            elif 'training' in keys: # LightGBM
                train_key = 'training'
                val_key = 'valid_1' if 'valid_1' in keys else keys[-1]
            elif 'validation_0' in keys: # XGBoost
                train_key = 'validation_0'
                val_key = 'validation_1'
            
            if not train_key or not val_key:
                print(f"Fold {i+1}: Could not detect standard keys. Found: {keys}")
                continue

            # Extract Metrics (Robust)
            # Try lowercase ('auc'), uppercase ('AUC'), and title case ('Auc')
            candidates = [metric, metric.upper(), metric.capitalize()]
            
            # Helper to safely get metric
            def get_data(source_dict, candidates):
                for c in candidates:
                    if c in source_dict:
                        return source_dict[c]
                return None

            # Get Data
            train_data = get_data(result[train_key], candidates)
            val_data = get_data(result[val_key], candidates)

            # Plotting
            lbl_val = 'Val' if i == 0 else None
            
            # If train metric is missing (common in CatBoost), just plot validation
            if train_data is None:
                if i == 0: print(f"Note: '{metric}' not found in Training log. Plotting Validation only.")
                plt.plot(val_data, color='red', alpha=0.6, linewidth=1.5, label=lbl_val)
            else:
                lbl_train = 'Train' if i == 0 else None
                plt.plot(train_data, color='blue', alpha=0.3, label=lbl_train)
                plt.plot(val_data, color='red', alpha=0.6, linewidth=1.5, label=lbl_val)

        plt.title(f'{self.model_name} Learning Curves ({metric.upper()})')
        plt.xlabel('Iterations')
        plt.ylabel(metric.upper())
        plt.legend()
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
            imp_dict = {}
            
            # Auto-Detect Importance Type
            try:
                # CatBoost
                if hasattr(model, 'get_feature_importance'):
                    imp = model.get_feature_importance()
                    names = model.feature_names_
                    imp_dict = dict(zip(names, imp))
                
                # LightGBM
                elif hasattr(model, 'feature_importances_') and hasattr(model, 'feature_name_'):
                    imp = model.feature_importances_
                    names = model.feature_name_
                    imp_dict = dict(zip(names, imp))
                
                # XGBoost (Booster object)
                elif hasattr(model, 'get_booster'):
                    imp_dict = model.get_booster().get_score(importance_type='gain')
                
                # Sklearn generic
                elif hasattr(model, 'feature_importances_'):
                     # Try to get names, otherwise use indices
                    names = getattr(model, 'feature_names_in_', [f'f{x}' for x in range(len(model.feature_importances_))])
                    imp_dict = dict(zip(names, model.feature_importances_))

            except Exception as e:
                print(f"Error extracting importance for model {i}: {e}")
                continue

            fold_imp = pd.DataFrame({
                'Feature': list(imp_dict.keys()),
                'Importance': list(imp_dict.values()),
                'Fold': i + 1
            })
            feature_importance = pd.concat([feature_importance, fold_imp], axis=0)

        if feature_importance.empty: return

        avg_imp = feature_importance.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(10, 12))
        sns.barplot(x=avg_imp.values, y=avg_imp.index, palette=self.palette)
        plt.title(f'{self.model_name} Top {top_n} Feature Importances (Average)')
        plt.xlabel('Importance')
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