"""
sensitivity_scorer.py
Module 1: Attribute Sensitivity Scorer
Detects proxy features for sensitive attributes using mutual information,
correlation analysis, and surrogate model proxy scoring.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


KNOWN_SENSITIVE_ATTRIBUTES = {
    'sex', 'gender', 'race', 'ethnicity', 'age', 'religion',
    'nationality', 'disability', 'marital_status', 'pregnancy',
    'national_origin', 'color', 'familial_status'
}

KNOWN_PROXIES = {
    'relationship': ['sex', 'gender'],
    'marital_status': ['sex', 'gender'],
    'occupation': ['sex', 'gender', 'race'],
    'zip_code': ['race', 'ethnicity'],
    'neighborhood': ['race', 'ethnicity'],
    'surname': ['race', 'ethnicity', 'nationality'],
    'first_name': ['sex', 'gender'],
    'hours_per_week': ['sex', 'gender'],
    'workclass': ['race', 'sex'],
}

DOMAIN_LEGAL_CONTEXT = {
    'hiring': {
        'laws': ['Title VII of the Civil Rights Act (42 U.S.C. 2000e)', 'EEOC Guidelines', 'EU AI Act Article 6 (High-Risk)', 'ADA (Americans with Disabilities Act)'],
        'protected_classes': ['sex', 'race', 'religion', 'national_origin', 'age', 'disability'],
        'disparate_impact_threshold': 0.80
    },
    'lending': {
        'laws': ['Equal Credit Opportunity Act (ECOA, 15 U.S.C. 1691)', 'Fair Housing Act', 'EU AI Act Article 6', 'CFPB Disparate Impact Rules'],
        'protected_classes': ['sex', 'race', 'religion', 'national_origin', 'marital_status', 'age', 'familial_status'],
        'disparate_impact_threshold': 0.80
    },
    'healthcare': {
        'laws': ['Section 1557 of the ACA', 'ADA Title III', 'EU AI Act Article 6', 'HIPAA Non-Discrimination'],
        'protected_classes': ['sex', 'race', 'disability', 'age', 'national_origin'],
        'disparate_impact_threshold': 0.85
    },
    'insurance': {
        'laws': ['State Insurance Codes', 'ACA Section 2704', 'EU AI Act Article 6'],
        'protected_classes': ['sex', 'race', 'disability', 'age'],
        'disparate_impact_threshold': 0.80
    }
}


def safe_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly convert all columns to float64.
    Handles Arrow-backed strings (Python 3.14 / Streamlit Cloud),
    object dtype strings, and category dtypes.
    """
    df_out = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col]
        # Convert to plain Python object dtype first to strip Arrow backing
        try:
            series = series.astype(object)
        except Exception:
            pass
        # If non-numeric, label encode
        try:
            df_out[col] = pd.to_numeric(series, errors='raise').astype(np.float64)
        except (ValueError, TypeError):
            le = LabelEncoder()
            df_out[col] = le.fit_transform(series.fillna('__missing__').astype(str)).astype(np.float64)
    return df_out


def encode_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Encode all columns to numeric, returning encoders for categorical cols."""
    df_enc = df.copy()
    encoders = {}
    for col in df_enc.columns:
        series = df_enc[col]
        # Strip Arrow backing
        try:
            series = series.astype(object)
        except Exception:
            pass
        is_str = (
            series.dtype == object
            or str(series.dtype) == 'string'
            or str(series.dtype).startswith('string')
            or hasattr(series.dtype, 'pyarrow_dtype')
            or series.dtype.name == 'category'
        )
        if is_str:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(series.fillna('__missing__').astype(str))
            encoders[col] = le
        else:
            try:
                df_enc[col] = pd.to_numeric(series, errors='coerce').fillna(0).astype(np.float64)
            except Exception:
                le = LabelEncoder()
                df_enc[col] = le.fit_transform(series.fillna('__missing__').astype(str))
                encoders[col] = le
    return df_enc, encoders


def compute_mutual_information(df: pd.DataFrame, target_col: str, sensitive_cols: list) -> pd.DataFrame:
    df_enc, _ = encode_dataframe(df)
    feature_cols = [c for c in df_enc.columns if c != target_col]

    # Use safe_to_float to handle any remaining Arrow-backed columns
    X = safe_to_float(df_enc[feature_cols].fillna(0))
    y_raw = df_enc[target_col].fillna(0)

    try:
        y = pd.to_numeric(y_raw.astype(object), errors='coerce').fillna(0).astype(np.float64)
    except Exception:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)).astype(np.float64), index=y_raw.index)

    is_classification = int(y.nunique()) <= 20

    if is_classification:
        mi_scores = mutual_info_classif(X.values, y.values, random_state=42)
    else:
        mi_scores = mutual_info_regression(X.values, y.values, random_state=42)

    mi_df = pd.DataFrame({'feature': feature_cols, 'mi_with_outcome': mi_scores})

    for sens in sensitive_cols:
        if sens not in df_enc.columns:
            continue
        y_sens_raw = df_enc[sens].fillna(0)
        try:
            y_sens = pd.to_numeric(y_sens_raw.astype(object), errors='coerce').fillna(0).astype(np.float64)
        except Exception:
            le = LabelEncoder()
            y_sens = pd.Series(le.fit_transform(y_sens_raw.astype(str)).astype(np.float64), index=y_sens_raw.index)

        other_features = [c for c in feature_cols if c != sens]
        X_other = safe_to_float(df_enc[other_features].fillna(0))

        if int(y_sens.nunique()) <= 20:
            proxy_scores = mutual_info_classif(X_other.values, y_sens.values, random_state=42)
        else:
            proxy_scores = mutual_info_regression(X_other.values, y_sens.values, random_state=42)

        temp = pd.DataFrame({'feature': other_features, f'mi_proxy_{sens}': proxy_scores})
        mi_df = mi_df.merge(temp, on='feature', how='left')

    return mi_df


def compute_proxy_score(df: pd.DataFrame, feature: str, sensitive_attr: str) -> float:
    """Train a surrogate model to predict the sensitive attribute from a single feature."""
    df_enc, _ = encode_dataframe(df[[feature, sensitive_attr]].dropna())
    X = safe_to_float(df_enc[[feature]])
    y_raw = df_enc[sensitive_attr]
    try:
        y = pd.to_numeric(y_raw.astype(object), errors='coerce').fillna(0)
    except Exception:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

    if int(y.nunique()) < 2:
        return 0.0

    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    try:
        scores = cross_val_score(model, X.values, y.values, cv=3, scoring='roc_auc')
        normalized = max(0.0, (scores.mean() - 0.5) / 0.5)
        return round(normalized, 4)
    except Exception:
        return 0.0


def compute_intersectional_risk(df: pd.DataFrame, feature_pairs: list, sensitive_cols: list, target_col: str) -> list:
    """Detect features that appear safe individually but are risky in combination."""
    df_enc, _ = encode_dataframe(df)
    results = []

    for f1, f2 in feature_pairs:
        if f1 not in df_enc.columns or f2 not in df_enc.columns:
            continue
        interaction = df_enc[f1].astype(str) + "_x_" + df_enc[f2].astype(str)
        le = LabelEncoder()
        interaction_enc = le.fit_transform(interaction)

        for sens in sensitive_cols:
            if sens not in df_enc.columns:
                continue
            y_sens_raw = df_enc[sens].fillna(0)
            try:
                y_sens = pd.to_numeric(y_sens_raw.astype(object), errors='coerce').fillna(0).astype(np.float64)
            except Exception:
                le_s = LabelEncoder()
                y_sens = pd.Series(le_s.fit_transform(y_sens_raw.astype(str)).astype(np.float64), index=y_sens_raw.index)

            if int(y_sens.nunique()) < 2:
                continue

            f1_arr = safe_to_float(df_enc[[f1]]).values
            f2_arr = safe_to_float(df_enc[[f2]]).values
            int_arr = interaction_enc.reshape(-1, 1)

            mi_f1 = mutual_info_classif(f1_arr, y_sens.values, random_state=42)[0]
            mi_f2 = mutual_info_classif(f2_arr, y_sens.values, random_state=42)[0]
            mi_combined = mutual_info_classif(int_arr, y_sens.values, random_state=42)[0]

            synergy = mi_combined - max(mi_f1, mi_f2)
            if synergy > 0.02:
                results.append({
                    'feature_1': f1,
                    'feature_2': f2,
                    'sensitive_attr': sens,
                    'mi_f1_alone': round(mi_f1, 4),
                    'mi_f2_alone': round(mi_f2, 4),
                    'mi_combined': round(mi_combined, 4),
                    'synergy_score': round(synergy, 4),
                    'blind_spot': mi_f1 < 0.05 and mi_f2 < 0.05 and mi_combined > 0.05
                })

    return sorted(results, key=lambda x: x['synergy_score'], reverse=True)


def run_sensitivity_analysis(
    df: pd.DataFrame,
    target_col: str,
    sensitive_cols: list,
    domain: str = 'hiring'
) -> dict:
    """Full Module 1 analysis pipeline."""

    # Defensive: convert entire dataframe object columns up front
    df = df.copy()
    for col in df.columns:
        try:
            df[col] = df[col].astype(object)
        except Exception:
            pass

    mi_df = compute_mutual_information(df, target_col, sensitive_cols)

    risk_records = []
    for _, row in mi_df.iterrows():
        feat = row['feature']
        if feat in sensitive_cols:
            continue

        proxy_cols = [c for c in mi_df.columns if c.startswith('mi_proxy_')]
        max_proxy_mi = max([row.get(c, 0) or 0 for c in proxy_cols]) if proxy_cols else 0

        known_proxy_targets = KNOWN_PROXIES.get(feat.lower().replace(' ', '_'), [])
        is_known_proxy = len(known_proxy_targets) > 0

        best_sens = None
        best_proxy_score = 0.0
        for sens in sensitive_cols:
            ps = compute_proxy_score(df, feat, sens)
            if ps > best_proxy_score:
                best_proxy_score = ps
                best_sens = sens

        composite = (
            0.35 * float(row.get('mi_with_outcome', 0) or 0)
            + 0.35 * float(max_proxy_mi or 0)
            + 0.20 * float(best_proxy_score or 0)
            + 0.10 * float(is_known_proxy)
        )

        proxy_target = best_sens or (known_proxy_targets[0] if known_proxy_targets else 'unknown')

        risk_records.append({
            'feature': feat,
            'mi_with_outcome': round(float(row.get('mi_with_outcome', 0) or 0), 4),
            'max_proxy_mi': round(float(max_proxy_mi or 0), 4),
            'surrogate_proxy_score': round(float(best_proxy_score or 0), 4),
            'is_known_proxy': is_known_proxy,
            'known_proxy_for': known_proxy_targets,
            'primary_proxy_target': proxy_target,
            'composite_risk_score': round(float(composite or 0), 4),
            'risk_tier': (
                'HIGH' if composite > 0.25
                else 'MEDIUM' if composite > 0.10
                else 'LOW'
            )
        })

    risk_df = pd.DataFrame(risk_records).sort_values('composite_risk_score', ascending=False)

    high_medium_features = risk_df[risk_df['risk_tier'].isin(['HIGH', 'MEDIUM'])]['feature'].tolist()[:6]
    pairs = [(high_medium_features[i], high_medium_features[j])
             for i in range(len(high_medium_features))
             for j in range(i+1, len(high_medium_features))]

    intersectional_risks = compute_intersectional_risk(df, pairs, sensitive_cols, target_col)
    legal_context = DOMAIN_LEGAL_CONTEXT.get(domain, DOMAIN_LEGAL_CONTEXT['hiring'])

    return {
        'risk_dataframe': risk_df,
        'intersectional_risks': intersectional_risks,
        'domain': domain,
        'legal_context': legal_context,
        'sensitive_cols': sensitive_cols,
        'target_col': target_col,
        'total_features_analyzed': len(risk_df),
        'high_risk_count': int((risk_df['risk_tier'] == 'HIGH').sum()),
        'medium_risk_count': int((risk_df['risk_tier'] == 'MEDIUM').sum()),
        'low_risk_count': int((risk_df['risk_tier'] == 'LOW').sum()),
    }
