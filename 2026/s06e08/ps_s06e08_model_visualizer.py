import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, mean_squared_error, r2_score
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance

class ModelVisualizer:
    """
    Centralizes all visualization logic for Gradient Boosting (LightGBM, XGBoost, CatBoost, HistGradientBoosting)
    and Neural Network models in PS-S06E08.
    """
    def __init__(self, palette='viridis', model_name=None):
        self.palette = palette
        self.model_name = model_name if model_name else 'Model'
        plt.style.use('seaborn-v0_8-whitegrid')  # Set modern whitegrid style

    
    def plot_learning_curves(self, eval_results, metric='auc', metric_period=1):
        """
        Plots Train vs Val metrics over iterations across cross-validation folds.
        Supports LightGBM, XGBoost, CatBoost, and HistGradientBoostingClassifier.
        """
        plt.figure(figsize=(10, 6))
        
        candidates = [
            metric, metric.upper(), metric.lower(),
            'mlogloss', 'multi_logloss', 'RMSE', 'rmse', 'RootMeanSquaredError', 
            'l2', 'auc', 'AUC', 'TotalF1', 'balanced_accuracy', 'loss', 'score'
        ]
        
        for i, result in enumerate(eval_results):
            # Case 1: HistGradientBoosting model object directly passed
            if hasattr(result, 'train_score_') and hasattr(result, 'validation_score_'):
                train_data = np.abs(result.train_score_)
                val_data = np.abs(result.validation_score_)
            elif isinstance(result, dict):
                keys = list(result.keys())
                train_key, val_key = None, None
                
                # Detect Keys based on framework
                if 'learn' in keys:  # CatBoost
                    train_key = 'learn'
                    val_keys = [k for k in keys if 'validation' in k]
                    val_key = val_keys[-1] if val_keys else None
                elif 'training' in keys:  # LightGBM
                    train_key = 'training'
                    val_key = 'valid_0' if 'valid_0' in keys else keys[-1]
                elif 'validation_0' in keys:  # XGBoost
                    train_key = 'validation_1' if 'validation_1' in keys else 'validation_0'
                    val_key = 'validation_0'
                elif 'train' in keys or 'training_loss' in keys:  # Generic / Scikit-Learn
                    train_key = 'train' if 'train' in keys else 'training_loss'
                    val_key = 'val' if 'val' in keys else 'valid_loss'
                
                if not train_key or not val_key:
                    if len(keys) >= 2:
                        train_key, val_key = keys[0], keys[1]
                    else:
                        print(f'Fold {i+1}: Could not detect standard keys. Found: {keys}')
                        continue
                        
                def get_data(source_dict, candidates):
                    if isinstance(source_dict, (np.ndarray, list)):
                        return source_dict
                    if isinstance(source_dict, dict):
                        for c in candidates:
                            if c in source_dict:
                                return source_dict[c]
                    return None
                    
                train_data = get_data(result[train_key], candidates)
                val_data = get_data(result[val_key], candidates)
                
                # Fallback: if train_data is None (e.g. CatBoost GPU missing AUC on learn), use common metric present in both learn and validation
                if train_data is None and isinstance(result[train_key], dict) and isinstance(result[val_key], dict):
                    common_keys = [k for k in result[train_key].keys() if k in result[val_key].keys()]
                    if common_keys:
                        train_data = result[train_key][common_keys[0]]
                        val_data = result[val_key][common_keys[0]]
            else:
                print(f'Fold {i+1}: Unsupported result type {type(result)}')
                continue

            lbl_train = 'Train' if i == 0 else None
            lbl_val = 'Val' if i == 0 else None
            
            if val_data is not None:
                total_iters = (len(val_data) - 1) * metric_period
                x_val = np.arange(len(val_data)) * metric_period
                plt.plot(x_val, val_data, color='red', alpha=0.6, linewidth=1.5, label=lbl_val)
            
            if train_data is not None:
                x_train = np.linspace(0, total_iters if val_data is not None else len(train_data), len(train_data))
                plt.plot(x_train, train_data, color='blue', alpha=0.3, label=lbl_train)    
                
        plt.title(f'{self.model_name} Learning Curves ({metric.upper()})')
        plt.xlabel('Iterations')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.show()

    
    def plot_feature_importance(self, models, X_val=None, y_val=None, top_n=30, show_values=True):
        """
        Aggregates and plots feature importance across multiple trained models.
        Supports CatBoost, LightGBM, XGBoost, and HistGradientBoostingClassifier (via Permutation Importance).
        """
        if not models:
            print('No models provided for feature importance.')
            return

        feature_importance = pd.DataFrame()
        
        for i, model_item in enumerate(models):
            # Unpack model if passed as (model, X, y) tuple
            if isinstance(model_item, tuple):
                model = model_item[0]
                fold_X_val = model_item[1] if len(model_item) > 1 else X_val
                fold_y_val = model_item[2] if len(model_item) > 2 else y_val
            else:
                model = model_item
                fold_X_val = X_val
                fold_y_val = y_val

            imp_dict = {}
            try:
                # CatBoost
                if hasattr(model, 'get_feature_importance'):
                    imp = model.get_feature_importance()
                    names = model.feature_names_ if hasattr(model, 'feature_names_') else [f'f{x}' for x in range(len(imp))]
                    imp_dict = dict(zip(names, imp))
                
                # LightGBM (Booster or Sklearn API)
                elif hasattr(model, 'feature_importances_'):
                    imp = model.feature_importances_
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
                elif hasattr(model, 'get_score'):
                    imp_dict = model.get_score(importance_type='gain')
                    
                # HistGradientBoosting & Scikit-Learn Estimators without tree feature_importances_
                elif fold_X_val is not None and fold_y_val is not None:
                    # Subsample if validation dataset is very large (> 10k rows) for performance
                    if len(fold_X_val) > 10000:
                        sample_idx = np.random.choice(len(fold_X_val), 10000, replace=False)
                        X_sub = fold_X_val.iloc[sample_idx] if hasattr(fold_X_val, 'iloc') else fold_X_val[sample_idx]
                        y_sub = fold_y_val.iloc[sample_idx] if hasattr(fold_y_val, 'iloc') else fold_y_val[sample_idx]
                    else:
                        X_sub, y_sub = fold_X_val, fold_y_val

                    perm_imp = permutation_importance(
                        model, X_sub, y_sub, 
                        scoring='roc_auc', n_repeats=5, random_state=42
                    )
                    feature_names = fold_X_val.columns if hasattr(fold_X_val, 'columns') else [f'f{x}' for x in range(X_sub.shape[1])]
                    imp_dict = dict(zip(feature_names, perm_imp.importances_mean))
                else:
                    if i == 0:
                        print(f"Note: {type(model).__name__} does not store tree feature_importances_ directly. Pass X_val and y_val to plot_feature_importance(models, X_val, y_val) to compute Permutation Importance.")
                    continue

            except Exception as e:
                print(f'Error extracting importance for model {i}: {e}')
                continue

            if imp_dict:
                fold_imp = pd.DataFrame({
                    'Feature': list(imp_dict.keys()),
                    'Importance': list(imp_dict.values()),
                    'Fold': i + 1
                })
                feature_importance = pd.concat([feature_importance, fold_imp], axis=0)

        if feature_importance.empty:
            return

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

    
    def plot_distribution_mismatch(self, y_true, y_preds):
        """
        Overlays the distribution of True values vs Predicted values.
        Crucial for detecting if the model is 'regressing to the mean'.
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(y_true, label='True Values', fill=True, color='blue', alpha=0.3, warn_singular=False)
        sns.kdeplot(y_preds, label='Predicted Values', fill=True, color='red', alpha=0.3, warn_singular=False)
        plt.title(f'{self.model_name}: Distribution Mismatch')
        plt.xlabel('Target Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    
    def plot_error_bias(self, y_true, y_preds):
        """
        Plots Error vs True Value. 
        Helps detect if the model fails specifically on high or low scores.
        """
        residuals = y_preds - y_true
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, residuals, alpha=0.5, color='teal', edgecolor='k')
        plt.axhline(0, color='red', linestyle='--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Prediction Error (Pred - True)')
        plt.title(f'{self.model_name}: Error Bias (Systematic Failures)')
        plt.grid(True, alpha=0.3)
        plt.show()

    
    def plot_training_heartbeat(self, history, title_suffix=''):
        """
        Plots Loss and Learning Rate side-by-side. 
        Essential for checking Cosine Annealing convergence.
        """
        if not history: return

        fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()
        
        # Plot Loss
        ax1.plot(history['train_loss'], label='Train Loss', color='tab:blue', alpha=0.6)
        if 'val_rmse' in history:
            ax1.plot(history['val_rmse'], label='Val RMSE', color='tab:orange', linewidth=2)
        
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('RMSE / Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)

        # Plot LR on secondary axis
        if 'lrs' in history:
            ax2 = ax1.twinx()
            ax2.plot(history['lrs'], label='Learning Rate', color='tab:red', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Learning Rate', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title(f'{self.model_name} Training Heartbeat {title_suffix}')
        
        # Combined Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = (ax2.get_legend_handles_labels()) if 'lrs' in history else ([], [])
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        
        plt.show()


    def plot_confusion_matrix(self, y_true, y_pred, classes=None, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        """
        Plots the confusion matrix.
        
        Args:
            y_true: Ground truth (correct) target values.
            y_pred: Estimated targets as returned by a classifier (class labels, not probabilities).
            classes: List of class names for the axis labels.
            normalize: If True, normalize the confusion matrix.
        """
        if not classes:
            classes = np.unique(y_true)

        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            print("Normalized confusion matrix")
        else:
            fmt = 'd'
            print('Confusion matrix, without normalization')

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=False,
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'{self.model_name}: {title}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    
    def plot_roc_curve(self, y_true, y_score, title='Receiver Operating Characteristic (ROC) Curve'):
        """
        Plots the ROC curve and calculates the AUC.
        
        Args:
            y_true: True binary labels.
            y_score: Target scores, can either be probability estimates of the positive class,
                     confidence values, or non-thresholded measure of decisions.
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        try:
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=self.model_name)
        except TypeError:
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=self.model_name)
        display.plot()
        
        plt.title(f'{self.model_name}: {title}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(alpha=0.3)
        plt.show()


    def plot_multiclass_roc_curve(self, y_true, y_score, classes=None, class_names=None, title='Multiclass ROC Curve (One-vs-Rest)'):
        """
        Plots the ROC curve and calculates the AUC for multiclass targets.
        
        Args:
            y_true: True labels.
            y_score: Target scores, can either be probability estimates of the positive class,
                     confidence values, or non-thresholded measure of decisions.
            classes: List of unique classes. If None, inferred from y_true.
            class_names: Dict mapping class label to string name. If None, labels are used.
        """
        if classes is None:
            classes = np.unique(y_true)
        if class_names is None:
            class_names = {c: str(c) for c in classes}
            
        # Binarize the output
        y_bin = label_binarize(y_true, classes=classes)
        n_classes = len(classes)
        
        plt.figure(figsize=(8, 6))
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            # If binary classification, y_bin might be 1D, so check shape
            y_true_class = y_bin[:, i] if n_classes > 2 else (y_true == classes[i]).astype(int)
            fpr, tpr, _ = roc_curve(y_true_class, y_score[:, i] if len(y_score.shape) > 1 else y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[classes[i]]} (AUC = {roc_auc:.2f})')
        
        plt.title(f'{self.model_name}: {title}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(alpha=0.3)
        plt.legend(loc='lower right')
        plt.show()
