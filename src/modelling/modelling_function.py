import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit
from src.feature_engineering import statistical_features as sf


def train_model(model, df: pd.DataFrame, drop_columns: list, target_column: str, n_splits: int = 5, early_stopping_rounds: int = 20):

    if 'timestamp' in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        
    to_drop = [col for col in drop_columns + [target_column] if col in df.columns]
    X = df.drop(columns=to_drop, errors='ignore')
    y = df[target_column]
    
    tscv = TimeSeriesSplit(n_splits=n_splits) #TSCV used to train and cross validate model on time series data, ensuring temporal order is preserved and preventing data leakage from future to past.
    
    fold = 1
    cv_metrics = []
    best_model = None
    results = None
    
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        
        zeros_count = (y_train == 0).sum()
        ones_count = (y_train == 1).sum()
        imbalance_ratio = zeros_count / ones_count if ones_count > 0 else 1 #calculate ratio because classes are not balanced

        if ones_count == 0:
            fold += 1
            continue

        model.set_params(
            scale_pos_weight=imbalance_ratio,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric='aucpr'
        )

            
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fold_aucpr = average_precision_score(y_val, y_pred_proba)
        cv_metrics.append(fold_aucpr)
        
        print(f"Fold {fold}/{n_splits} | Iteracje: {model.best_iteration:4d} | AUCPR walidacyjne: {fold_aucpr:.4f}")
        
        if fold == n_splits:
            best_model = model
            results = model.evals_result()
            final_X_train_columns = X_train.columns
            
        fold += 1

    print(f"\nŚrednie AUCPR z {n_splits} foldów: {np.mean(cv_metrics):.4f}")
    
    plot_learning_curves(results)
    plot_feature_importance(best_model, final_X_train_columns)
            
    joblib.dump(best_model, 'trained_model.joblib')
    print("Model saved")    
    return best_model

def plot_learning_curves(results):
    plt.figure(figsize=(10, 5))
    metric = list(results['validation_0'].keys())[0]
    plt.plot(results['validation_0'][metric], label='Train')
    plt.plot(results['validation_1'][metric], label='Validation', linestyle='--')
    plt.title(f'Learning Curves ({metric})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(model, feature_names):
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    plt.barh(range(len(indices)), importances[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('Feature Importance')
    plt.show()



def evaluate_model(model, test_df, drop_columns, target_column, threshold=0.5):
    to_drop = [col for col in drop_columns + [target_column] if col in test_df.columns]
    X_test = test_df.drop(columns=to_drop, errors='ignore')
    y_test = test_df[target_column]
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int) 
    
    print(classification_report(y_test, y_pred, zero_division=0))
    
    if len(np.unique(y_test)) > 1:
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"PR AUC (AUCPR) Score: {average_precision_score(y_test, y_pred_proba):.4f}")
    else:
        print("ROC AUC Score: N/A")
        print("PR AUC (AUCPR) Score: N/A")
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Incident']).plot(cmap='Blues')
    plt.show()



def predict(model, df_window, value_column="value", W=12):
    
    df = df_window.copy()
    df = sf.add_lag(df, value_column, W)
    df = sf.add_mean_std_W(df, value_column, W)
    df = sf.add_relative_features(df, value_column)
    df = sf.add_advanced_features(df, value_column, W)
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    latest_row = df.iloc[-1:]
    
    
    X_latest = latest_row.drop(columns=['timestamp', 'target', 'is_incident'], errors='ignore')    
    
    
    return model.predict_proba(X_latest)[0, 1]