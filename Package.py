# ══════════════════════════════════════════════════════════════════════════════
# Package.py
# Trains the final Gradient Boosting model and saves a self-contained package
# (.pkl) that the Streamlit app can load directly.
#
# Run ONCE:   python Package.py
# Output:     GradBoost_package.pkl  (saved in the 2 Gradboost folder)
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score
)

# ── 1. Load data ───────────────────────────────────────────────────────────────
DATA_PATH = "C:/Reseach/Kim project 3/Data/Data5/4_Thermal_Fixed.xlsx"

data = pd.read_excel(DATA_PATH)

TARGET = 'H2O2 conc. (mg/L)'
X = data.drop(TARGET, axis=1)
y = data[TARGET]

# Categorical columns by position (same indices used across all models)
categorical_features = [0, 1, 4, 5, 6, 8, 12]
categorical_cols = list(X.columns[categorical_features])
numerical_cols   = list(X.columns.drop(categorical_cols))
feature_names    = list(X.columns)
k                = X.shape[1]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Collect unique category values for the UI drop-downs
cat_values = {col: sorted(X[col].dropna().unique().tolist())
              for col in categorical_cols}

# Data-driven stats for the numeric inputs (defaults / slider ranges in the app)
num_stats = {
    col: {
        'min'   : float(X[col].min()),
        'max'   : float(X[col].max()),
        'median': float(X[col].median()),
    }
    for col in numerical_cols
}

# ── 2. Train / Test split (same seed as your analysis script) ─────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

print(f"Total  : {len(X)}")
print(f"Train  : {len(X_train)}")
print(f"Test   : {len(X_test)}")

# ── 3. Hard-coded best hyperparameters (from your Optuna run) ─────────────────
BEST_PARAMS = dict(
    n_estimators  = 878,
    learning_rate = 0.02981454415805585,
    max_depth     = 5,
    subsample     = 0.7403007051781213,
    random_state  = 42,
)

def make_model():
    # Encoding done INSIDE the pipeline (fit per fold -> no leakage).
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(**BEST_PARAMS))
    ])

# ── 4. Metric helpers ──────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def rsr(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true)

def compute_metrics(y_true, y_pred, k):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n    = len(y_true)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        'R2':     r2,
        'AR2':    adjusted_r2(r2, n, k),
        'EVS':    explained_variance_score(y_true, y_pred),
        'RMSE':   rmse,
        'CVRMSE': rmse / np.mean(y_true) * 100,
        'MAPE':   mape(y_true, y_pred),
        'RSR':    rsr(y_true, y_pred),
        'MAE':    mean_absolute_error(y_true, y_pred),
        'RAE':    (np.sum(np.abs(y_true - y_pred)) /
                   np.sum(np.abs(y_true - np.mean(y_true)))),
        'SMAPE':  smape(y_true, y_pred),
    }

# ── 5. 5-Fold CV (validation metrics) ─────────────────────────────────────────
N_OUTER = 5
outer_cv = KFold(n_splits=N_OUTER, shuffle=True, random_state=42)

fold_metrics_val   = []
fold_metrics_train = []
oof_preds   = np.full(len(y_train), np.nan)
oof_actuals = np.full(len(y_train), np.nan)

for fold, (tr_idx, val_idx) in enumerate(outer_cv.split(X_train)):
    X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
    X_vl = X_train.iloc[val_idx].reset_index(drop=True)
    y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
    y_vl = y_train.iloc[val_idx].reset_index(drop=True)

    m = make_model()
    m.fit(X_tr, y_tr)

    vl_preds = m.predict(X_vl)
    tr_preds = m.predict(X_tr)

    oof_preds[val_idx]   = vl_preds
    oof_actuals[val_idx] = y_vl.values

    fold_metrics_val.append(compute_metrics(y_vl.values, vl_preds, k))
    fold_metrics_train.append(compute_metrics(y_tr.values, tr_preds, k))
    print(f"Fold {fold+1} | Val R2: {fold_metrics_val[-1]['R2']:.4f}  RMSE: {fold_metrics_val[-1]['RMSE']:.4f}")

# Aggregate
metric_names = list(fold_metrics_val[0].keys())
cv_results = {}
for mn in metric_names:
    cv_results[mn] = {
        'train_mean': np.mean([f[mn] for f in fold_metrics_train]),
        'train_std' : np.std( [f[mn] for f in fold_metrics_train]),
        'val_mean'  : np.mean([f[mn] for f in fold_metrics_val]),
        'val_std'   : np.std( [f[mn] for f in fold_metrics_val]),
    }

# ── 6. Final model on X_train ─────────────────────────────────────────────────
print("\nTraining final model on X_train...")
final_model = make_model()
final_model.fit(X_train, y_train)

train_preds_final = final_model.predict(X_train)
test_preds_final  = final_model.predict(X_test)

train_metrics_final = compute_metrics(y_train.values, train_preds_final, k)
test_metrics_final  = compute_metrics(y_test.values,  test_preds_final,  k)

# ── 7. Print summary ───────────────────────────────────────────────────────────
print("\n" + "="*80)
print("METRICS SUMMARY")
print("="*80)
print(f"{'Metric':<10} {'Train (CV)':>20} {'Val (CV)':>20} {'Test (Held-out)':>16}")
print("-"*80)
for mn in metric_names:
    r = cv_results[mn]
    print(f"{mn:<10} {r['train_mean']:>10.4f} +/- {r['train_std']:<7.4f} "
          f"{r['val_mean']:>10.4f} +/- {r['val_std']:<7.4f} "
          f"{test_metrics_final[mn]:>16.4f}")

# ── 8. Feature importances ────────────────────────────────────────────────────
# The pipeline one-hot-expands the categorical columns, so importances are given
# on the transformed features. Aggregate them back to the original columns:
#   * numerical column   -> its single importance
#   * categorical column -> sum of importances over all its one-hot categories
preprocessor  = final_model.named_steps['preprocessor']
gbr           = final_model.named_steps['model']
transformed   = list(preprocessor.get_feature_names_out())
raw_imports   = gbr.feature_importances_

feat_importances = {f: 0.0 for f in feature_names}
for t_name, imp in zip(transformed, raw_imports):
    # t_name looks like 'cat__<col>_<category>' or 'num__<col>'
    body = t_name.split('__', 1)[1]
    matched = None
    for col in feature_names:
        if body == col or body.startswith(col + '_'):
            # prefer the longest matching column name (handles overlaps)
            if matched is None or len(col) > len(matched):
                matched = col
    if matched is not None:
        feat_importances[matched] += float(imp)

# ── 9. Save package ───────────────────────────────────────────────────────────
package = {
    # Model
    'model'                : final_model,
    'best_params'          : BEST_PARAMS,

    # Data schema
    'target'               : TARGET,
    'feature_names'        : feature_names,
    'categorical_cols'     : categorical_cols,
    'numerical_cols'       : numerical_cols,
    'cat_values'           : cat_values,
    'num_stats'            : num_stats,

    # Performance metrics (for display in the app)
    'cv_results'           : cv_results,
    'train_metrics_final'  : train_metrics_final,
    'test_metrics_final'   : test_metrics_final,

    # OOF predictions (for scatter plot in app)
    'oof_preds'            : oof_preds,
    'oof_actuals'          : oof_actuals,
    'y_train'              : y_train.values,
    'train_preds_final'    : train_preds_final,
    'y_test'               : y_test.values,
    'test_preds_final'     : test_preds_final,

    # Feature importances
    'feat_importances'     : feat_importances,
}

PKG_PATH = "C:/Reseach/Kim project 3/ML Codes/2 Gradboost (Boosting)/GradBoost_package.pkl"
joblib.dump(package, PKG_PATH, compress=3)
print(f"\n[OK] Package saved -> {PKG_PATH}")
