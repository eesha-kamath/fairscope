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


def _strip_arrow(series: pd.Series) -> pd.Series:
    """Strip Arrow backing from pandas Series (Python 3.14 / Streamlit Cloud)."""
    try:
        return series.astype(object)
    except Exception:
        return series


def _safe_encode_value(encoders: dict, col: str, val):
    """Encode a single value using stored encoders, with fallback."""
    if col not in encoders:
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0
    try:
        return float(encoders[col].transform([str(val)])[0])
    except (ValueError, KeyError):
        return 0.0


def _encode_row_to_array(row: pd.Series, feature_cols: list, encoders: dict) -> np.ndarray:
    """Encode a single dataframe row into a float64 numpy array."""
    result = []
    for col in feature_cols:
        raw = row.get(col, 0)
        if col in encoders:
            val = _safe_encode_value(encoders, col, raw)
        else:
            try:
                val = float(raw)
            except (ValueError, TypeError):
                val = 0.0
        result.append(val)
    return np.array(result, dtype=np.float64).reshape(1, -1)


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
    """Run a single counterfactual: change one feature, observe outcome change."""
    base_arr = _encode_row_to_array(base_record, feature_cols, encoders)
    cf_arr = base_arr.copy()

    # Find index of feature to change
    if feature_to_change in feature_cols:
        idx = feature_cols.index(feature_to_change)
        cf_arr[0, idx] = _safe_encode_value(encoders, feature_to_change, counterfactual_value)

    base_pred = int(model.predict(base_arr)[0])
    cf_pred   = int(model.predict(cf_arr)[0])
    base_prob = float(model.predict_proba(base_arr)[0][1])
    cf_prob   = float(model.predict_proba(cf_arr)[0][1])
    prob_delta = cf_prob - base_prob

    return {
        'feature_changed': feature_to_change,
        'original_value': original_value,
        'counterfactual_value': counterfactual_value,
        'original_prediction': base_pred,
        'counterfactual_prediction': cf_pred,
        'original_probability': round(base_prob, 4),
        'counterfactual_probability': round(cf_prob, 4),
        'probability_delta': round(prob_delta, 4),
        'outcome_changed': base_pred != cf_pred,
        'bias_signal': abs(prob_delta) > 0.05 or base_pred != cf_pred
    }


def run_multifeature_counterfactual(
    model,
    base_record: pd.Series,
    df_reference: pd.DataFrame,
    feature_changes: list,
    feature_cols: list,
    encoders: dict
) -> dict:
    """Change multiple features simultaneously to test combined proxy effects."""
    base_arr = _encode_row_to_array(base_record, feature_cols, encoders)
    cf_arr = base_arr.copy()

    for change in feature_changes:
        feat = change['feature']
        new_val = change['changed']
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            cf_arr[0, idx] = _safe_encode_value(encoders, feat, new_val)

    base_pred = int(model.predict(base_arr)[0])
    cf_pred   = int(model.predict(cf_arr)[0])
    base_prob = float(model.predict_proba(base_arr)[0][1])
    cf_prob   = float(model.predict_proba(cf_arr)[0][1])

    return {
        'feature_changes': feature_changes,
        'original_prediction': base_pred,
        'counterfactual_prediction': cf_pred,
        'original_probability': round(base_prob, 4),
        'counterfactual_probability': round(cf_prob, 4),
        'probability_delta': round(float(cf_prob - base_prob), 4),
        'outcome_changed': base_pred != cf_pred,
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
    2. For each, run counterfactuals changing high-risk proxy features
    3. Aggregate how often outcome changes and by how much
    """
    # Strip Arrow backing from the entire dataframe upfront
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = _strip_arrow(df_clean[col])

    sample_size = min(n_samples, len(df_clean))
    sample_df = df_clean.sample(sample_size, random_state=42)

    all_results = []
    feature_impact_summary = {}

    for feat in high_risk_features[:6]:
        if feat not in df_clean.columns:
            continue

        unique_vals = df_clean[feat].dropna().unique().tolist()
        if len(unique_vals) < 2:
            continue

        val_pairs = list(itertools.combinations(unique_vals[:4], 2))[:3]
        feat_outcomes_changed = 0
        feat_prob_deltas = []

        for _, row in sample_df.iterrows():
            for val_a, val_b in val_pairs:
                try:
                    result = run_counterfactual_test(
                        model, row, df_clean,
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
        mean_delta = float(np.mean(feat_prob_deltas)) if feat_prob_deltas else 0.0
        change_rate = feat_outcomes_changed / max(total_tests, 1)

        feature_impact_summary[feat] = {
            'outcome_change_rate': round(change_rate, 4),
            'mean_prob_delta': round(mean_delta, 4),
            'max_prob_delta': round(max(feat_prob_deltas) if feat_prob_deltas else 0.0, 4),
            'bias_detected': change_rate > 0.10 or mean_delta > 0.05
        }

    # Multi-feature combined probe
    combined_probe_results = []
    sensitive_linked = [f for f in high_risk_features if f in feature_impact_summary][:3]

    if len(sensitive_linked) >= 2 and sensitive_cols:
        for _, row in sample_df.head(20).iterrows():
            changes = []
            for feat in sensitive_linked:
                unique_vals = df_clean[feat].dropna().unique().tolist()
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
                        model, row, df_clean, changes, feature_cols, encoders
                    )
                    combined_probe_results.append(result)
                except Exception:
                    continue

    combined_bias_rate = (
        sum(1 for r in combined_probe_results if r['outcome_changed']) /
        max(len(combined_probe_results), 1)
    )
    combined_mean_delta = (
        float(np.mean([abs(r['probability_delta']) for r in combined_probe_results]))
        if combined_probe_results else 0.0
    )

    return {
        'feature_impact_summary': feature_impact_summary,
        'all_single_feature_results': all_results,
        'combined_probe_results': combined_probe_results,
        'combined_bias_rate': round(combined_bias_rate, 4),
        'combined_mean_prob_delta': round(combined_mean_delta, 4),
        'total_tests_run': len(all_results) + len(combined_probe_results),
        'overall_bias_detected': (
            any(v['bias_detected'] for v in feature_impact_summary.values()) or
            combined_bias_rate > 0.10
        ),
        'high_risk_features_tested': list(feature_impact_summary.keys())
    }