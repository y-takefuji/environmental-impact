import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cluster import FeatureAgglomeration
from xgboost import XGBRegressor
import shap
from scipy.stats import spearmanr

# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================

df = pd.read_csv('final_raw_sample_0_percent.csv')

all_cols = df.columns.tolist()
target_col = 'Total Environmental Cost'

string_cols = all_cols[1:5]

pct_intensity_cols = [
    'Total Environmental Intensity (Revenue)',
    'Total Environmental Intensity (Operating Income)'
]

last_col = '% Imputed'

def parse_indian_number(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val == '' or val == '-' or val.lower() == 'nan':
        return np.nan
    negative = False
    if val.startswith('(') and val.endswith(')'):
        negative = True
        val = val[1:-1]
    val = val.replace(',', '')
    try:
        num = float(val)
        return -num if negative else num
    except:
        return np.nan

def parse_percentage(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val == '' or val == '-' or val.lower() == 'nan':
        return np.nan
    val = val.replace('%', '')
    try:
        return float(val)
    except:
        return np.nan

numeric_cols = [c for c in all_cols if c not in string_cols and c != target_col]
for col in numeric_cols:
    if col in pct_intensity_cols or col == last_col:
        df[col] = df[col].apply(parse_percentage)
    else:
        df[col] = df[col].apply(parse_indian_number)

df[target_col] = df[target_col].apply(parse_indian_number)
df.fillna(0, inplace=True)

# ============================================================
# 2. PREPARE FEATURE MATRIX
# ============================================================

feature_cols = [c for c in all_cols if c not in string_cols and c != target_col]
X_full = df[feature_cols].copy()
y = df[target_col].copy()

print("=" * 60)
print("DATA SHAPE")
print("=" * 60)
print(f"Full DataFrame shape : {df.shape}")
print(f"Feature matrix shape : {X_full.shape}")
print(f"Target vector shape  : {y.shape}")

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
print(y.describe())
print(f"\nSkewness : {y.skew():.4f}")
print(f"Kurtosis : {y.kurt():.4f}")
print(f"Min      : {y.min():.4f}")
print(f"Max      : {y.max():.4f}")
print(f"Zeros    : {(y == 0).sum()}")

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def cv_r2_rf(X, y, n=5):
    model = RandomForestRegressor(random_state=42)
    scores = cross_val_score(model, X, y, cv=n, scoring='r2')
    return round(float(np.mean(scores)), 4)

def cv_r2_xgb(X, y, n=5):
    model = XGBRegressor(random_state=42, verbosity=0)
    scores = cross_val_score(model, X, y, cv=n, scoring='r2')
    return round(float(np.mean(scores)), 4)


def fa_feature_scores(X_data, n_clusters):
    """
    Fit FeatureAgglomeration, then score every individual feature
    by the variance of that feature's own raw values — no mean,
    no cluster-signal aggregation.
    Features are ranked globally across ALL clusters in one list;
    no per-cluster top picking is performed.
    The cluster structure from FA determines grouping only;
    each feature keeps its own individual variance as its score.
    """
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X_data)
    labels = fa.labels_
    X_arr  = X_data.values.astype(float)
    names  = list(X_data.columns)

    scores = np.array([
        np.var(X_arr[:, i])
        for i in range(X_arr.shape[1])
    ])

    print(f"  FA cluster assignments: { {n: int(labels[i]) for i, n in enumerate(names)} }")
    return scores


def fa_select_top(X_data, n_top):
    """
    Use fa_feature_scores to rank ALL features globally across ALL clusters,
    then return the top-n feature names by descending variance score.
    n_clusters = max(5, sqrt(n_features)) as default.
    """
    n_clusters = max(5, int(np.sqrt(X_data.shape[1])))
    print(f"  FA n_clusters = {n_clusters}, n_features = {X_data.shape[1]}")
    scores = fa_feature_scores(X_data, n_clusters)
    names  = list(X_data.columns)
    # rank globally across ALL clusters by descending variance
    ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    top_features = [name for name, _ in ranked[:n_top]]
    print(f"  FA global variance scores (top {n_top}):")
    for name, sc in ranked[:n_top]:
        print(f"    {name}: {sc:.6f}")
    return top_features


# ============================================================
# 4. FEATURE SELECTION — RF
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION: RF")
print("=" * 60)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_full, y)
rf_imp = pd.Series(rf_model.feature_importances_, index=feature_cols)
rf_top5 = rf_imp.nlargest(5).index.tolist()
print(f"Top 5 features : {rf_top5}")

rf_cv5 = cv_r2_rf(X_full[rf_top5], y)
print(f"CV5 R²         : {rf_cv5}")

# Remove highest, reselect top 4
rf_highest = rf_top5[0]
X_reduced_rf = X_full.drop(columns=[rf_highest])
rf_model2 = RandomForestRegressor(random_state=42)
rf_model2.fit(X_reduced_rf, y)
rf_imp2 = pd.Series(rf_model2.feature_importances_, index=X_reduced_rf.columns)
rf_top4 = rf_imp2.nlargest(4).index.tolist()
print(f"Top 4 (reduced): {rf_top4}")

# ============================================================
# 5. FEATURE SELECTION — RF-SHAP
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION: RF-SHAP")
print("=" * 60)

rf_shap_model = RandomForestRegressor(random_state=42)
rf_shap_model.fit(X_full, y)

np.random.seed(42)
sample_idx = np.random.choice(len(X_full), size=min(100, len(X_full)), replace=False)
X_sample = X_full.iloc[sample_idx]

explainer_rf = shap.TreeExplainer(rf_shap_model)
shap_values_rf = explainer_rf.shap_values(X_sample)
rf_shap_imp = pd.Series(np.abs(shap_values_rf).mean(axis=0), index=feature_cols)
rf_shap_top5 = rf_shap_imp.nlargest(5).index.tolist()
print(f"Top 5 features : {rf_shap_top5}")

rf_shap_cv5 = cv_r2_rf(X_full[rf_shap_top5], y)
print(f"CV5 R²         : {rf_shap_cv5}")

# Remove highest, reselect top 4
rf_shap_highest = rf_shap_top5[0]
X_reduced_rfshap = X_full.drop(columns=[rf_shap_highest])
rf_shap_model2 = RandomForestRegressor(random_state=42)
rf_shap_model2.fit(X_reduced_rfshap, y)
explainer_rf2 = shap.TreeExplainer(rf_shap_model2)
shap_values_rf2 = explainer_rf2.shap_values(X_reduced_rfshap.iloc[sample_idx])
rf_shap_imp2 = pd.Series(np.abs(shap_values_rf2).mean(axis=0), index=X_reduced_rfshap.columns)
rf_shap_top4 = rf_shap_imp2.nlargest(4).index.tolist()
print(f"Top 4 (reduced): {rf_shap_top4}")

# ============================================================
# 6. FEATURE SELECTION — XGB
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION: XGB")
print("=" * 60)

xgb_model = XGBRegressor(random_state=42, verbosity=0)
xgb_model.fit(X_full, y)
xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols)
xgb_top5 = xgb_imp.nlargest(5).index.tolist()
print(f"Top 5 features : {xgb_top5}")

xgb_cv5 = cv_r2_xgb(X_full[xgb_top5], y)
print(f"CV5 R²         : {xgb_cv5}")

# Remove highest, reselect top 4
xgb_highest = xgb_top5[0]
X_reduced_xgb = X_full.drop(columns=[xgb_highest])
xgb_model2 = XGBRegressor(random_state=42, verbosity=0)
xgb_model2.fit(X_reduced_xgb, y)
xgb_imp2 = pd.Series(xgb_model2.feature_importances_, index=X_reduced_xgb.columns)
xgb_top4 = xgb_imp2.nlargest(4).index.tolist()
print(f"Top 4 (reduced): {xgb_top4}")

# ============================================================
# 7. FEATURE SELECTION — XGB-SHAP
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION: XGB-SHAP")
print("=" * 60)

xgb_shap_model = XGBRegressor(random_state=42, verbosity=0)
xgb_shap_model.fit(X_full, y)

X_sample_xgb = X_full.iloc[sample_idx]
explainer_xgb = shap.TreeExplainer(xgb_shap_model)
shap_values_xgb = explainer_xgb.shap_values(X_sample_xgb)
xgb_shap_imp = pd.Series(np.abs(shap_values_xgb).mean(axis=0), index=feature_cols)
xgb_shap_top5 = xgb_shap_imp.nlargest(5).index.tolist()
print(f"Top 5 features : {xgb_shap_top5}")

xgb_shap_cv5 = cv_r2_xgb(X_full[xgb_shap_top5], y)
print(f"CV5 R²         : {xgb_shap_cv5}")

# Remove highest, reselect top 4
xgb_shap_highest = xgb_shap_top5[0]
X_reduced_xgbshap = X_full.drop(columns=[xgb_shap_highest])
xgb_shap_model2 = XGBRegressor(random_state=42, verbosity=0)
xgb_shap_model2.fit(X_reduced_xgbshap, y)
explainer_xgb2 = shap.TreeExplainer(xgb_shap_model2)
shap_values_xgb2 = explainer_xgb2.shap_values(X_reduced_xgbshap.iloc[sample_idx])
xgb_shap_imp2 = pd.Series(np.abs(shap_values_xgb2).mean(axis=0), index=X_reduced_xgbshap.columns)
xgb_shap_top4 = xgb_shap_imp2.nlargest(4).index.tolist()
print(f"Top 4 (reduced): {xgb_shap_top4}")

# ============================================================
# 8. FEATURE SELECTION — PURE FA
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION: Pure Feature Agglomeration (FA)")
print("=" * 60)

# Full dataset: top 5
fa_top5 = fa_select_top(X_full, n_top=5)
print(f"Top 5 features : {fa_top5}")

fa_cv5 = cv_r2_rf(X_full[fa_top5], y)
print(f"CV5 R²         : {fa_cv5}")

# Remove highest-ranked feature, reselect top 4 from reduced dataset
fa_highest = fa_top5[0]
print(f"\n  Removing highest FA feature: '{fa_highest}'")
X_reduced_fa = X_full.drop(columns=[fa_highest])

fa_top4 = fa_select_top(X_reduced_fa, n_top=4)
print(f"Top 4 (reduced): {fa_top4}")

# ============================================================
# 9. FEATURE SELECTION — HVGS (Variance-based)
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION: HVGS (Variance-based)")
print("=" * 60)

hvgs_var = X_full.var()
hvgs_top5 = hvgs_var.nlargest(5).index.tolist()
print(f"Top 5 features : {hvgs_top5}")

hvgs_cv5 = cv_r2_rf(X_full[hvgs_top5], y)
print(f"CV5 R²         : {hvgs_cv5}")

# Remove highest, reselect top 4
hvgs_highest = hvgs_top5[0]
X_reduced_hvgs = X_full.drop(columns=[hvgs_highest])
hvgs_var2 = X_reduced_hvgs.var()
hvgs_top4 = hvgs_var2.nlargest(4).index.tolist()
print(f"Top 4 (reduced): {hvgs_top4}")

# ============================================================
# 10. FEATURE SELECTION — SPEARMAN
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION: Spearman Correlation")
print("=" * 60)

spearman_scores = {
    col: abs(spearmanr(X_full[col], y)[0])
    for col in feature_cols
}
spearman_series = pd.Series(spearman_scores)
spearman_top5 = spearman_series.nlargest(5).index.tolist()
print(f"Top 5 features : {spearman_top5}")

spearman_cv5 = cv_r2_rf(X_full[spearman_top5], y)
print(f"CV5 R²         : {spearman_cv5}")

# Remove highest, reselect top 4
spearman_highest = spearman_top5[0]
X_reduced_sp = X_full.drop(columns=[spearman_highest])
spearman_scores2 = {
    col: abs(spearmanr(X_reduced_sp[col], y)[0])
    for col in X_reduced_sp.columns
}
spearman_series2 = pd.Series(spearman_scores2)
spearman_top4 = spearman_series2.nlargest(4).index.tolist()
print(f"Top 4 (reduced): {spearman_top4}")

# ============================================================
# 11. SUMMARY TABLE
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)

def fmt(feat_list):
    return ' | '.join(feat_list)

summary = pd.DataFrame({
    'Method': ['RF', 'RF-SHAP', 'XGB', 'XGB-SHAP', 'FA', 'HVGS', 'Spearman'],
    'CV5 Accuracy (R²)': [
        rf_cv5, rf_shap_cv5, xgb_cv5, xgb_shap_cv5,
        fa_cv5, hvgs_cv5, spearman_cv5
    ],
    'Top 5 Features': [
        fmt(rf_top5), fmt(rf_shap_top5), fmt(xgb_top5), fmt(xgb_shap_top5),
        fmt(fa_top5), fmt(hvgs_top5), fmt(spearman_top5)
    ],
    'Top 4 Features (Reduced)': [
        fmt(rf_top4), fmt(rf_shap_top4), fmt(xgb_top4), fmt(xgb_shap_top4),
        fmt(fa_top4), fmt(hvgs_top4), fmt(spearman_top4)
    ]
})

print(summary.to_string(index=False))
summary.to_csv('result.csv', index=False)
print("\nSaved summary to result.csv")
