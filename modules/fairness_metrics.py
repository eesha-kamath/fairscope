"""
fairness_metrics.py
Module 2: Fairness Conflict Visualizer
Computes standard fairness metrics, detects conflicts between them,
and quantifies accuracy-fairness trade-offs.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


METRIC_DESCRIPTIONS = {
    'demographic_parity_difference': {
        'name': 'Demographic Parity Difference',
        'short': 'DPD',
        'definition': 'The difference in positive prediction rates between privileged and unprivileged groups. A value of 0 means equal positive rates across groups.',
        'ideal': 0.0,
        'threshold': 0.1,
        'legal_basis': 'Supported by EEOC Four-Fifths Rule for adverse impact; EU AI Act recital on non-discrimination.',
        'conflicts_with': ['equalized_odds_difference', 'equal_opportunity_difference'],
        'tradeoff_note': 'Enforcing DPD = 0 often reduces accuracy when base rates differ across groups.'
    },
    'equalized_odds_difference': {
        'name': 'Equalized Odds Difference',
        'short': 'EOD',
        'definition': 'Requires equal True Positive Rate AND equal False Positive Rate across groups simultaneously.',
        'ideal': 0.0,
        'threshold': 0.1,
        'legal_basis': 'Aligned with equal treatment doctrine; used in criminal justice fairness standards (COMPAS debates).',
        'conflicts_with': ['demographic_parity_difference', 'calibration_difference'],
        'tradeoff_note': 'Often incompatible with Demographic Parity when group base rates differ (Chouldechova, 2017).'
    },
    'equal_opportunity_difference': {
        'name': 'Equal Opportunity Difference',
        'short': 'EQOPD',
        'definition': 'The difference in True Positive Rate (recall) between groups. Focuses on who benefits from positive predictions.',
        'ideal': 0.0,
        'threshold': 0.1,
        'legal_basis': 'Aligns with disparate impact doctrine under Title VII and ECOA for positive outcome distribution.',
        'conflicts_with': ['demographic_parity_difference'],
        'tradeoff_note': 'More lenient than Equalized Odds; allows FPR to differ across groups.'
    },
    'predictive_parity_difference': {
        'name': 'Predictive Parity Difference',
        'short': 'PPD',
        'definition': 'The difference in Positive Predictive Value (precision) across groups. Measures if predictions are equally accurate per group.',
        'ideal': 0.0,
        'threshold': 0.1,
        'legal_basis': 'Used in calibration arguments; relevant to lending (CFPB) where score accuracy per group matters.',
        'conflicts_with': ['equalized_odds_difference'],
        'tradeoff_note': 'Mathematically proven incompatible with Equalized Odds when base rates differ (Kleinberg, 2016).'
    },
    'calibration_difference': {
        'name': 'Calibration Difference',
        'short': 'CAL',
        'definition': 'Whether predicted probabilities reflect true outcomes equally across groups (e.g., 70% prediction means 70% actual positive rate in each group).',
        'ideal': 0.0,
        'threshold': 0.05,
        'legal_basis': 'Critical for healthcare AI (ACA Section 1557); ensures model reliability per group.',
        'conflicts_with': ['equalized_odds_difference'],
        'tradeoff_note': 'A well-calibrated model may still show disparate impact in outcomes.'
    }
}

CONFLICT_MATRIX = {
    ('demographic_parity_difference', 'equalized_odds_difference'): {
        'conflict_type': 'MATHEMATICAL',
        'description': 'When group base rates differ, achieving DPD = 0 requires accepting EOD > 0, and vice versa. This is a proven mathematical impossibility (Chouldechova, 2017).',
        'severity': 'HIGH'
    },
    ('equalized_odds_difference', 'predictive_parity_difference'): {
        'conflict_type': 'MATHEMATICAL',
        'description': 'Cannot simultaneously achieve Equalized Odds and Predictive Parity when base rates differ across groups (Kleinberg et al., 2016; Chouldechova, 2017).',
        'severity': 'HIGH'
    },
    ('demographic_parity_difference', 'equal_opportunity_difference'): {
        'conflict_type': 'PRACTICAL',
        'description': 'Optimizing for equal positive rates (DPD) often conflicts with equal recall (EQOPD) unless groups have identical base rates.',
        'severity': 'MEDIUM'
    },
    ('equalized_odds_difference', 'calibration_difference'): {
        'conflict_type': 'MATHEMATICAL',
        'description': 'A perfectly calibrated model will generally not satisfy Equalized Odds unless groups have equal base rates.',
        'severity': 'HIGH'
    }
}


def _strip_arrow(series: pd.Series) -> pd.Series:
    """Strip Arrow backing from a pandas Series so dtype operations work on Python 3.14."""
    try:
        return series.astype(object)
    except Exception:
        return series


def _to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a Series to float64, label-encoding if it contains strings."""
    series = _strip_arrow(series)
    try:
        return pd.to_numeric(series, errors='raise').astype(np.float64)
    except (ValueError, TypeError):
        le = LabelEncoder()
        return pd.Series(
            le.fit_transform(series.fillna('__missing__').astype(str)).astype(np.float64),
            index=series.index
        )


def _encode_df_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert every column in df to float64, safely handling Arrow-backed
    string columns (Python 3.14 / Streamlit Cloud) and all other dtypes.
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        out[col] = _to_numeric_series(df[col])
    return out


def encode_features(df: pd.DataFrame, target_col: str, sensitive_col: str):
    """Encode all columns, return X, y, s as numpy-ready float64 DataFrames."""
    df_work = df.copy()

    # Strip Arrow backing from every column first
    for col in df_work.columns:
        df_work[col] = _strip_arrow(df_work[col])

    encoders = {}
    for col in df_work.columns:
        series = df_work[col]
        is_non_numeric = (
            series.dtype == object
            or str(series.dtype).startswith('string')
            or series.dtype.name == 'category'
        )
        if is_non_numeric:
            le = LabelEncoder()
            df_work[col] = le.fit_transform(series.fillna('__missing__').astype(str))
            encoders[col] = le
        else:
            df_work[col] = pd.to_numeric(series, errors='coerce').fillna(0)

    feature_cols = [c for c in df_work.columns if c not in [target_col, sensitive_col]]
    X = _encode_df_to_float(df_work[feature_cols].fillna(0))
    y = _to_numeric_series(df_work[target_col].fillna(0))
    s = _to_numeric_series(df_work[sensitive_col].fillna(0))

    return X, y, s, encoders


def train_base_model(X_train, y_train):
    # Ensure pure numpy arrays reach sklearn
    X_np = np.array(X_train, dtype=np.float64)
    y_np = np.array(y_train, dtype=np.float64)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_np, y_np)
    return model


def compute_group_metrics(y_true, y_pred, y_prob, group_mask):
    gt = np.array(y_true)[group_mask]
    gp = np.array(y_pred)[group_mask]
    gprob = np.array(y_prob)[group_mask]

    if len(gt) == 0:
        return {}

    tp = int(((gp == 1) & (gt == 1)).sum())
    fp = int(((gp == 1) & (gt == 0)).sum())
    fn = int(((gp == 0) & (gt == 1)).sum())
    tn = int(((gp == 0) & (gt == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    base_rate = float(gt.mean())
    cal = float(abs(gprob.mean() - base_rate)) if len(gprob) > 1 else 0.0

    return {
        'n': int(len(gt)),
        'positive_rate': round(float(gp.mean()), 4),
        'true_positive_rate': round(tpr, 4),
        'false_positive_rate': round(fpr, 4),
        'positive_predictive_value': round(ppv, 4),
        'base_rate': round(base_rate, 4),
        'calibration_error': round(cal, 4),
        'accuracy': round(float(accuracy_score(gt, gp)), 4)
    }


def compute_all_fairness_metrics(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    privileged_group_value=None
) -> dict:

    X, y, s, encoders = encode_features(df, target_col, sensitive_col)

    # Convert everything to plain numpy before sklearn
    X_np = X.values.astype(np.float64)
    y_np = y.values.astype(np.float64)
    s_np = s.values.astype(np.float64)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_np, y_np, s_np, test_size=0.3, random_state=42,
        stratify=y_np
    )

    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    overall_accuracy = accuracy_score(y_test, y_pred)
    overall_auc = roc_auc_score(y_test, y_prob)

    unique_groups = np.unique(s_test)
    group_counts = pd.Series(s_test).value_counts()

    if privileged_group_value is not None and sensitive_col in encoders:
        try:
            priv_encoded = float(encoders[sensitive_col].transform([str(privileged_group_value)])[0])
        except Exception:
            priv_encoded = float(group_counts.idxmax())
        privileged_group = priv_encoded
    else:
        privileged_group = float(group_counts.idxmax())

    unprivileged_groups = [g for g in unique_groups if g != privileged_group]

    if sensitive_col in encoders:
        le = encoders[sensitive_col]
        group_label_map = {
            float(le.transform([cls])[0]): cls
            for cls in le.classes_
        }
    else:
        group_label_map = {float(g): str(g) for g in unique_groups}

    group_metrics = {}
    for g in unique_groups:
        mask = (s_test == g)
        label = group_label_map.get(float(g), str(g))
        group_metrics[label] = compute_group_metrics(y_test, y_pred, y_prob, mask)

    priv_label = group_label_map.get(float(privileged_group), str(privileged_group))
    priv_metrics = group_metrics.get(priv_label, {})

    fairness_metrics = {}
    for unpriv_g in unprivileged_groups:
        unpriv_label = group_label_map.get(float(unpriv_g), str(unpriv_g))
        unpriv_metrics = group_metrics.get(unpriv_label, {})
        if not unpriv_metrics or not priv_metrics:
            continue
        fairness_metrics[unpriv_label] = {
            'demographic_parity_difference': round(
                priv_metrics.get('positive_rate', 0) - unpriv_metrics.get('positive_rate', 0), 4),
            'equalized_odds_difference': round(max(
                abs(priv_metrics.get('true_positive_rate', 0) - unpriv_metrics.get('true_positive_rate', 0)),
                abs(priv_metrics.get('false_positive_rate', 0) - unpriv_metrics.get('false_positive_rate', 0))
            ), 4),
            'equal_opportunity_difference': round(
                priv_metrics.get('true_positive_rate', 0) - unpriv_metrics.get('true_positive_rate', 0), 4),
            'predictive_parity_difference': round(
                abs(priv_metrics.get('positive_predictive_value', 0) - unpriv_metrics.get('positive_predictive_value', 0)), 4),
            'calibration_difference': round(
                abs(priv_metrics.get('calibration_error', 0) - unpriv_metrics.get('calibration_error', 0)), 4)
        }

    if fairness_metrics:
        aggregate_metrics = {}
        for metric in list(fairness_metrics.values())[0].keys():
            vals = [fairness_metrics[g][metric] for g in fairness_metrics]
            aggregate_metrics[metric] = round(max(vals, key=abs), 4)
    else:
        aggregate_metrics = {m: 0.0 for m in METRIC_DESCRIPTIONS}

    detected_conflicts = []
    metric_keys = list(aggregate_metrics.keys())
    for i in range(len(metric_keys)):
        for j in range(i+1, len(metric_keys)):
            pair = (metric_keys[i], metric_keys[j])
            pair_rev = (metric_keys[j], metric_keys[i])
            conflict = CONFLICT_MATRIX.get(pair) or CONFLICT_MATRIX.get(pair_rev)
            if conflict:
                m1_val = abs(aggregate_metrics[metric_keys[i]])
                m2_val = abs(aggregate_metrics[metric_keys[j]])
                if (m1_val > METRIC_DESCRIPTIONS[metric_keys[i]]['threshold'] or
                        m2_val > METRIC_DESCRIPTIONS[metric_keys[j]]['threshold']):
                    detected_conflicts.append({
                        'metric_1': metric_keys[i],
                        'metric_2': metric_keys[j],
                        'metric_1_value': aggregate_metrics[metric_keys[i]],
                        'metric_2_value': aggregate_metrics[metric_keys[j]],
                        **conflict
                    })

    tradeoffs = estimate_accuracy_tradeoffs(
        model, X_train, y_train, X_test, y_test, s_test,
        aggregate_metrics, overall_accuracy
    )

    return {
        'model': model,
        'feature_cols': list(X.columns),
        'encoders': encoders,
        'overall_accuracy': round(overall_accuracy, 4),
        'overall_auc': round(overall_auc, 4),
        'group_metrics': group_metrics,
        'fairness_metrics': fairness_metrics,
        'aggregate_metrics': aggregate_metrics,
        'detected_conflicts': detected_conflicts,
        'tradeoffs': tradeoffs,
        'privileged_group': priv_label,
        'unprivileged_groups': [group_label_map.get(float(g), str(g)) for g in unprivileged_groups],
        'sensitive_col': sensitive_col,
        'target_col': target_col,
        'n_test': len(y_test),
        'X_test': X_test,
        'y_test': y_test,
        's_test': s_test,
        'group_label_map': group_label_map
    }


def estimate_accuracy_tradeoffs(model, X_train, y_train, X_test, y_test, s_test, aggregate_metrics, baseline_accuracy):
    tradeoffs = {}
    X_test_np = np.array(X_test, dtype=np.float64)
    y_test_np = np.array(y_test, dtype=np.float64)
    s_test_np = np.array(s_test, dtype=np.float64)

    y_prob = model.predict_proba(X_test_np)[:, 1]
    mode_group = float(pd.Series(s_test_np).mode()[0])

    for metric_name, metric_val in aggregate_metrics.items():
        if abs(metric_val) < 0.01:
            tradeoffs[metric_name] = {'accuracy_cost_pct': 0.0, 'feasible': True}
            continue

        best_cost = 99.0
        for threshold_adj in np.linspace(-0.3, 0.3, 30):
            y_pred_adj = np.where(
                s_test_np == mode_group,
                (y_prob > 0.5).astype(int),
                (y_prob > (0.5 + threshold_adj)).astype(int)
            )
            cost = (baseline_accuracy - accuracy_score(y_test_np, y_pred_adj)) * 100
            if 0 <= cost < best_cost:
                best_cost = cost

        tradeoffs[metric_name] = {
            'accuracy_cost_pct': round(best_cost, 2),
            'feasible': best_cost < 10.0
        }

    return tradeoffs