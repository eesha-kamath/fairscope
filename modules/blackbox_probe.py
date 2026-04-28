"""
blackbox_probe.py
Module 3: Black-Box Probe
Counterfactual testing to audit models via API or locally.
Tests if changing only sensitive-attribute-correlated features changes outcomes,
revealing hidden discrimination without access to model internals.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import itertools
import warnings
warnings.filterwarnings('ignore')


COUNTERFACTUAL_TEMPLATES = {
    'hiring': {
        'description': 'Tests if identical candidate profiles receive different hiring scores based on inferred demographic signals.',
        'sensitive_signals': {
            'sex': {
                'marital_status': {'Male': 'Married-civ-spouse', 'Female': 'Divorced'},
                'relationship': {'Male': 'Husband', 'Female': 'Wife'},
                'occupation': {'Male': 'Craft-repair', 'Female': 'Other-service'},
            },
            'race': {
                'workclass': {'White': 'Self-emp-inc', 'Black': 'Private'},
            }
        }
    },
    'lending': {
        'description': 'Tests if identical loan applications receive different scores based on neighborhood or name proxies.',
        'sensitive_signals': {
            'sex': {
                'marital_status': {'Male': 'Single', 'Female': 'Married'},
            },
            'race': {
                'zip_code': {'White': '10001', 'Black': '60619'},
            }
        }
    }
}


def encode_single_row(row: pd.Series, df_reference: pd.DataFrame) -> pd.Series:
    """Encode a single row using reference dataframe label encoders."""
    row_encoded = row.copy()
    for col in df_reference.columns:
        if df_reference[col].dtype == object or df_reference[col].dtype.name == 'category':
            le = LabelEncoder()
            le.fit(df_reference[col].astype(str))
            try:
                row_encoded[col] = le.transform([str(row[col])])[0]
            except ValueError:
                row_encoded[col] = 0
    return row_encoded


def run_counterfactual_test(
    model,
    base_record: pd.Series,
    df_reference: pd.DataFrame,
    feature_to_change: str,
    original_value,
    counterfactual_value,
    feature_cols: list,
    encoders: dict
) -> dict:
    """
    Run a single counterfactual: change one feature, observe outcome change.
    """
    # Encode base record
    base_encoded = {}
    for col in feature_cols:
        val = base_record.get(col, 0)
        if col in encoders:
            try:
                val = encoders[col].transform([str(val)])[0]
            except (ValueError, KeyError):
                val = 0
        base_encoded[col] = val

    # Create counterfactual record
    cf_encoded = base_encoded.copy()
    if feature_to_change in encoders:
        try:
            cf_encoded[feature_to_change] = encoders[feature_to_change].transform([str(counterfactual_value)])[0]
        except (ValueError, KeyError):
            cf_encoded[feature_to_change] = 0
    else:
        cf_encoded[feature_to_change] = counterfactual_value

    base_df = pd.DataFrame([base_encoded])[feature_cols]
    cf_df = pd.DataFrame([cf_encoded])[feature_cols]

    base_pred = model.predict(base_df)[0]
    cf_pred = model.predict(cf_df)[0]
    base_prob = model.predict_proba(base_df)[0][1]
    cf_prob = model.predict_proba(cf_df)[0][1]

    outcome_changed = base_pred != cf_pred
    prob_delta = cf_prob - base_prob

    return {
        'feature_changed': feature_to_change,
        'original_value': original_value,
        'counterfactual_value': counterfactual_value,
        'original_prediction': int(base_pred),
        'counterfactual_prediction': int(cf_pred),
        'original_probability': round(float(base_prob), 4),
        'counterfactual_probability': round(float(cf_prob), 4),
        'probability_delta': round(float(prob_delta), 4),
        'outcome_changed': bool(outcome_changed),
        'bias_signal': abs(prob_delta) > 0.05 or outcome_changed
    }


def run_multifeature_counterfactual(
    model,
    base_record: pd.Series,
    df_reference: pd.DataFrame,
    feature_changes: list,
    feature_cols: list,
    encoders: dict
) -> dict:
    """
    Change multiple features simultaneously to test combined proxy effects.
    feature_changes: [{'feature': str, 'original': val, 'changed': val}, ...]
    """
    base_encoded = {}
    for col in feature_cols:
        val = base_record.get(col, 0)
        if col in encoders:
            try:
                val = encoders[col].transform([str(val)])[0]
            except (ValueError, KeyError):
                val = 0
        base_encoded[col] = val

    cf_encoded = base_encoded.copy()
    for change in feature_changes:
        feat = change['feature']
        new_val = change['changed']
        if feat in encoders:
            try:
                cf_encoded[feat] = encoders[feat].transform([str(new_val)])[0]
            except (ValueError, KeyError):
                cf_encoded[feat] = 0
        else:
            cf_encoded[feat] = new_val

    base_df = pd.DataFrame([base_encoded])[feature_cols]
    cf_df = pd.DataFrame([cf_encoded])[feature_cols]

    base_pred = model.predict(base_df)[0]
    cf_pred = model.predict(cf_df)[0]
    base_prob = model.predict_proba(base_df)[0][1]
    cf_prob = model.predict_proba(cf_df)[0][1]

    return {
        'feature_changes': feature_changes,
        'original_prediction': int(base_pred),
        'counterfactual_prediction': int(cf_pred),
        'original_probability': round(float(base_prob), 4),
        'counterfactual_probability': round(float(cf_prob), 4),
        'probability_delta': round(float(cf_prob - base_prob), 4),
        'outcome_changed': bool(base_pred != cf_pred),
        'bias_signal': abs(cf_prob - base_prob) > 0.05 or base_pred != cf_pred
    }


def run_systematic_probe(
    model,
    df: pd.DataFrame,
    target_col: str,
    sensitive_cols: list,
    high_risk_features: list,
    feature_cols: list,
    encoders: dict,
    n_samples: int = 50
) -> dict:
    """
    Systematic black-box audit:
    1. Sample N records from the dataset
    2. For each record, run counterfactuals changing high-risk proxy features
    3. Aggregate how often outcome changes, and by how much
    """
    df_features = df.drop(columns=[target_col] + [c for c in sensitive_cols if c in df.columns], errors='ignore')

    sample_size = min(n_samples, len(df))
    sample_df = df.sample(sample_size, random_state=42)

    all_results = []
    feature_impact_summary = {}

    for feat in high_risk_features[:6]:
        if feat not in df.columns:
            continue

        unique_vals = df[feat].dropna().unique()
        if len(unique_vals) < 2:
            continue

        val_pairs = list(itertools.combinations(unique_vals[:4], 2))[:3]
        feat_outcomes_changed = 0
        feat_prob_deltas = []

        for _, row in sample_df.iterrows():
            for val_a, val_b in val_pairs:
                try:
                    result = run_counterfactual_test(
                        model, row, df,
                        feat, val_a, val_b,
                        feature_cols, encoders
                    )
                    all_results.append(result)
                    if result['outcome_changed']:
                        feat_outcomes_changed += 1
                    feat_prob_deltas.append(abs(result['probability_delta']))
                except Exception:
                    continue

        total_tests = sample_size * len(val_pairs)
        feature_impact_summary[feat] = {
            'outcome_change_rate': round(feat_outcomes_changed / max(total_tests, 1), 4),
            'mean_prob_delta': round(np.mean(feat_prob_deltas) if feat_prob_deltas else 0.0, 4),
            'max_prob_delta': round(max(feat_prob_deltas) if feat_prob_deltas else 0.0, 4),
            'bias_detected': (feat_outcomes_changed / max(total_tests, 1)) > 0.10 or
                             (np.mean(feat_prob_deltas) if feat_prob_deltas else 0) > 0.05
        }

    # Multi-feature combined probe (simulate full demographic flip)
    combined_probe_results = []
    sensitive_linked_features = [f for f in high_risk_features if f in feature_impact_summary][:3]

    if len(sensitive_linked_features) >= 2 and sensitive_cols:
        for _, row in sample_df.head(20).iterrows():
            changes = []
            for feat in sensitive_linked_features:
                unique_vals = df[feat].dropna().unique()
                if len(unique_vals) >= 2:
                    original = row.get(feat)
                    other_vals = [v for v in unique_vals if v != original]
                    if other_vals:
                        changes.append({
                            'feature': feat,
                            'original': original,
                            'changed': other_vals[0]
                        })

            if len(changes) >= 2:
                try:
                    result = run_multifeature_counterfactual(
                        model, row, df, changes, feature_cols, encoders
                    )
                    combined_probe_results.append(result)
                except Exception:
                    continue

    combined_bias_rate = sum(1 for r in combined_probe_results if r['outcome_changed']) / max(len(combined_probe_results), 1)
    combined_mean_delta = np.mean([abs(r['probability_delta']) for r in combined_probe_results]) if combined_probe_results else 0.0

    return {
        'feature_impact_summary': feature_impact_summary,
        'all_single_feature_results': all_results,
        'combined_probe_results': combined_probe_results,
        'combined_bias_rate': round(float(combined_bias_rate), 4),
        'combined_mean_prob_delta': round(float(combined_mean_delta), 4),
        'total_tests_run': len(all_results) + len(combined_probe_results),
        'overall_bias_detected': any(v['bias_detected'] for v in feature_impact_summary.values()) or combined_bias_rate > 0.10,
        'high_risk_features_tested': list(feature_impact_summary.keys())
    }
