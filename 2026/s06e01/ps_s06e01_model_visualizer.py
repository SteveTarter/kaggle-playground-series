import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

class ModelVisualizer:
    """
    Centralizes all visualization logic for Gradient Boosting models (XGBoost/CatBoost).
    """
    def __init__(self, palette='viridis', model_name=None):
        print('In ModelVisualizer __init__...')
        
        self.palette = palette
        self.model_name = model_name if model_name else "Model"
        plt.style.use('seaborn-v0_8-whitegrid') # Set a nice style

    def plot_learning_curves(self, eval_results, metric='rmse'):
        """
        Plots Train vs Val metrics over iterations.
        Handles case sensitivity ('RMSE' vs 'rmse') and missing training metrics.
        """
        plt.figure(figsize=(10, 6))
        
        # Extended candidates for Regression metrics
        candidates = [
            metric, metric.upper(), metric.lower(),
            'RMSE', 'rmse', 'RootMeanSquaredError', 'l2'
        ]
        
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
                val_key = 'valid_0' if 'valid_0' in keys else keys[-1]
            elif 'validation_0' in keys: # XGBoost
                train_key = 'validation_1'
                val_key = 'validation_0'
            
            if not train_key or not val_key:
                # Fallback for simpler structures
                if len(keys) >= 2:
                    train_key, val_key = keys[0], keys[1]
                else:
                    print(f"Fold {i+1}: Could not detect standard keys. Found: {keys}")
                    continue
                    
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
            lbl_train = 'Train' if i == 0 else None
            
            if val_data is not None:
                plt.plot(val_data, color='red', alpha=0.6, linewidth=1.5, label=lbl_val)
            
            if train_data is not None:
                plt.plot(train_data, color='blue', alpha=0.3, label=lbl_train)  
                
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
            try:
                # CatBoost
                if hasattr(model, 'get_feature_importance'):
                    imp = model.get_feature_importance()
                    names = model.feature_names_
                    imp_dict = dict(zip(names, imp))
                
                # LightGBM (Booster or Sklearn API)
                elif hasattr(model, 'feature_importances_'):
                    imp = model.feature_importances_
                    # Try to get names, handle different attribute names
                    if hasattr(model, 'feature_name_'):
                        names = model.feature_name_
                    elif hasattr(model, 'feature_names_in_'):
                        names = model.feature_names_in_
                    else:
                        names = [f'f{x}' for x in range(len(imp))]
                    imp_dict = dict(zip(names, imp))
                    
                # XGBoost (Booster object)
                elif hasattr(model, 'get_booster'):
                    imp_dict = model.get_booster().get_score(importance_type='gain')
                
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

        # Calculate mean importance
        avg_imp = feature_importance.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(10, len(avg_imp) * 0.4))
        
        ax = sns.barplot(x=avg_imp.values, y=avg_imp.index, palette=self.palette)

        if show_values:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

            xmax = ax.get_xlim()[1]
            ax.set_xlim(0, xmax * 1.15)
            
        plt.title(f'{self.model_name} Top {top_n} Feature Importances (Average)')
        plt.xlabel('Importance')
        plt.show()


    def plot_prediction_error(self, y_true, y_preds):
        """
        Plots True vs Predicted values.
        Replaces ROC Curve for Regression tasks.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_preds))
        r2 = r2_score(y_true, y_preds)
        
        plt.figure(figsize=(8, 8))

        # Scatter plot of predictions
        sns.scatterplot(x=y_true, y=y_preds, alpha=0.5, color='blue', edgecolor='k')
        
        # Perfect prediction line (Identity line)
        min_val = min(y_true.min(), y_preds.min())
        max_val = max(y_true.max(), y_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.title(f'{self.model_name} Prediction Error\nRMSE: {rmse:.4f} | R2: {r2:.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def plot_residuals(self, y_true, y_preds):
        """
        Plots Residuals (True - Pred) vs Predicted values.
        Helps diagnose bias and heteroscedasticity.
        """
        residuals = y_true - y_preds
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_preds, y=residuals, alpha=0.5, color='purple', edgecolor='k')
        plt.axhline(0, color='red', linestyle='--', lw=2)
        
        plt.title(f'{self.model_name} Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals (True - Pred)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()