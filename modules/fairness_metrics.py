"""
fairness_metrics.py
Module 2: Fairness Conflict Visualizer
Computes standard fairness metrics, detects conflicts between them,
and quantifies accuracy-fairness trade-offs.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


def encode_features(df: pd.DataFrame, target_col: str, sensitive_col: str) -> tuple:
    df_work = df.copy()
    encoders = {}

    for col in df_work.columns:
        if df_work[col].dtype == object or df_work[col].dtype.name == 'category':
            le = LabelEncoder()
            df_work[col] = le.fit_transform(df_work[col].astype(str))
            encoders[col] = le

    feature_cols = [c for c in df_work.columns if c not in [target_col, sensitive_col]]
    X = df_work[feature_cols].fillna(0)
    y = df_work[target_col].fillna(0)
    s = df_work[sensitive_col].fillna(0)

    return X, y, s, encoders


def train_base_model(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, group_mask: np.ndarray) -> dict:
    group_true = y_true[group_mask]
    group_pred = y_pred[group_mask]
    group_prob = y_prob[group_mask]

    if len(group_true) == 0:
        return {}

    positive_rate = group_pred.mean()
    tp = ((group_pred == 1) & (group_true == 1)).sum()
    fp = ((group_pred == 1) & (group_true == 0)).sum()
    fn = ((group_pred == 0) & (group_true == 1)).sum()
    tn = ((group_pred == 0) & (group_true == 0)).sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    base_rate = group_true.mean()

    # Calibration: correlation between predicted prob and actual outcome
    if len(group_prob) > 1:
        cal = np.abs(group_prob.mean() - base_rate)
    else:
        cal = 0.0

    return {
        'n': int(len(group_true)),
        'positive_rate': round(float(positive_rate), 4),
        'true_positive_rate': round(float(tpr), 4),
        'false_positive_rate': round(float(fpr), 4),
        'positive_predictive_value': round(float(ppv), 4),
        'base_rate': round(float(base_rate), 4),
        'calibration_error': round(float(cal), 4),
        'accuracy': round(float(accuracy_score(group_true, group_pred)), 4)
    }


def compute_all_fairness_metrics(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    privileged_group_value: str = None
) -> dict:
    X, y, s, encoders = encode_features(df, target_col, sensitive_col)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=0.3, random_state=42, stratify=y
    )

    model = train_base_model(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    overall_accuracy = accuracy_score(y_test, y_pred)
    overall_auc = roc_auc_score(y_test, y_prob)

    # Identify privileged group
    unique_groups = s_test.unique()
    group_counts = pd.Series(s_test).value_counts()
    if privileged_group_value is not None:
        if sensitive_col in encoders:
            priv_encoded = encoders[sensitive_col].transform([str(privileged_group_value)])[0]
        else:
            priv_encoded = privileged_group_value
        privileged_group = priv_encoded
    else:
        privileged_group = group_counts.idxmax()

    unprivileged_groups = [g for g in unique_groups if g != privileged_group]

    # Get group label names
    if sensitive_col in encoders:
        group_label_map = {v: k for k, v in zip(encoders[sensitive_col].classes_, encoders[sensitive_col].transform(encoders[sensitive_col].classes_))}
    else:
        group_label_map = {g: str(g) for g in unique_groups}

    # Compute per-group metrics
    group_metrics = {}
    for g in unique_groups:
        mask = (s_test == g).values
        group_metrics[group_label_map.get(g, str(g))] = compute_group_metrics(
            y_test.values, y_pred, y_prob, mask
        )

    priv_label = group_label_map.get(privileged_group, str(privileged_group))
    priv_metrics = group_metrics.get(priv_label, {})

    # Compute fairness metric differences (privileged vs worst unprivileged)
    fairness_metrics = {}
    for unpriv_g in unprivileged_groups:
        unpriv_label = group_label_map.get(unpriv_g, str(unpriv_g))
        unpriv_metrics = group_metrics.get(unpriv_label, {})
        if not unpriv_metrics or not priv_metrics:
            continue

        fairness_metrics[unpriv_label] = {
            'demographic_parity_difference': round(
                priv_metrics.get('positive_rate', 0) - unpriv_metrics.get('positive_rate', 0), 4
            ),
            'equalized_odds_difference': round(
                max(
                    abs(priv_metrics.get('true_positive_rate', 0) - unpriv_metrics.get('true_positive_rate', 0)),
                    abs(priv_metrics.get('false_positive_rate', 0) - unpriv_metrics.get('false_positive_rate', 0))
                ), 4
            ),
            'equal_opportunity_difference': round(
                priv_metrics.get('true_positive_rate', 0) - unpriv_metrics.get('true_positive_rate', 0), 4
            ),
            'predictive_parity_difference': round(
                abs(priv_metrics.get('positive_predictive_value', 0) - unpriv_metrics.get('positive_predictive_value', 0)), 4
            ),
            'calibration_difference': round(
                abs(priv_metrics.get('calibration_error', 0) - unpriv_metrics.get('calibration_error', 0)), 4
            )
        }

    # Aggregate worst-case differences across all unprivileged groups
    if fairness_metrics:
        aggregate_metrics = {}
        for metric in list(fairness_metrics.values())[0].keys():
            vals = [fairness_metrics[g][metric] for g in fairness_metrics]
            aggregate_metrics[metric] = round(max(vals, key=abs), 4)
    else:
        aggregate_metrics = {m: 0.0 for m in METRIC_DESCRIPTIONS}

    # Detect conflicts
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
                if m1_val > METRIC_DESCRIPTIONS[metric_keys[i]]['threshold'] or \
                   m2_val > METRIC_DESCRIPTIONS[metric_keys[j]]['threshold']:
                    detected_conflicts.append({
                        'metric_1': metric_keys[i],
                        'metric_2': metric_keys[j],
                        'metric_1_value': aggregate_metrics[metric_keys[i]],
                        'metric_2_value': aggregate_metrics[metric_keys[j]],
                        **conflict
                    })

    # Accuracy-fairness trade-off: estimate cost of enforcing each metric
    tradeoffs = estimate_accuracy_tradeoffs(model, X_train, y_train, X_test, y_test, s_test, aggregate_metrics, overall_accuracy)

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
        'unprivileged_groups': [group_label_map.get(g, str(g)) for g in unprivileged_groups],
        'sensitive_col': sensitive_col,
        'target_col': target_col,
        'n_test': len(y_test),
        'X_test': X_test,
        'y_test': y_test,
        's_test': s_test,
        'group_label_map': group_label_map
    }


def estimate_accuracy_tradeoffs(model, X_train, y_train, X_test, y_test, s_test, aggregate_metrics, baseline_accuracy) -> dict:
    """
    Estimate accuracy cost of enforcing each fairness criterion via threshold adjustment.
    """
    tradeoffs = {}
    y_prob = model.predict_proba(X_test)[:, 1]
    unique_groups = s_test.unique()

    for metric_name, metric_val in aggregate_metrics.items():
        if abs(metric_val) < 0.01:
            tradeoffs[metric_name] = {'accuracy_cost_pct': 0.0, 'feasible': True}
            continue

        # Simulate threshold adjustment per group to equalize metric
        best_cost = 99.0
        for threshold_adj in np.linspace(-0.3, 0.3, 30):
            y_pred_adjusted = []
            for i, (prob, group) in enumerate(zip(y_prob, s_test)):
                if group == s_test.mode()[0]:
                    y_pred_adjusted.append(1 if prob > 0.5 else 0)
                else:
                    y_pred_adjusted.append(1 if prob > (0.5 + threshold_adj) else 0)

            adj_accuracy = accuracy_score(y_test, y_pred_adjusted)
            cost = (baseline_accuracy - adj_accuracy) * 100
            if 0 <= cost < best_cost:
                best_cost = cost

        tradeoffs[metric_name] = {
            'accuracy_cost_pct': round(best_cost, 2),
            'feasible': best_cost < 10.0
        }

    return tradeoffs
