"""
visualizations.py
All Plotly chart builders for Fairscope's dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


COLORS = {
    'HIGH': '#C82828',
    'MEDIUM': '#C88200',
    'LOW': '#0A9650',
    'primary': '#0F2850',
    'accent': '#1A6FBF',
    'light': '#E8EFF8',
    'grid': '#E0E0E0',
    'text': '#2A2A2A',
    'pass': '#0A9650',
    'fail': '#C82828',
    'neutral': '#555555'
}

LAYOUT_BASE = dict(
    font=dict(family='IBM Plex Mono, monospace', color=COLORS['text'], size=11),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=40, b=20),
)


def plot_sensitivity_bar(risk_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    df = risk_df.head(top_n).copy()
    df = df.sort_values('composite_risk_score', ascending=True)

    color_map = {'HIGH': COLORS['HIGH'], 'MEDIUM': COLORS['MEDIUM'], 'LOW': COLORS['LOW']}
    colors = [color_map.get(t, COLORS['neutral']) for t in df['risk_tier']]

    fig = go.Figure(go.Bar(
        y=df['feature'],
        x=df['composite_risk_score'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in df['composite_risk_score']],
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Composite Risk: %{x:.4f}<br>"
            "<extra></extra>"
        )
    ))

    fig.add_vline(x=0.25, line_dash='dot', line_color=COLORS['HIGH'], line_width=1.5,
                  annotation_text='HIGH threshold', annotation_position='top right',
                  annotation_font_size=9)
    fig.add_vline(x=0.10, line_dash='dot', line_color=COLORS['MEDIUM'], line_width=1.5,
                  annotation_text='MEDIUM threshold', annotation_position='bottom right',
                  annotation_font_size=9)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text='Feature Proxy Risk Scores', font=dict(size=13, color=COLORS['primary'])),
        xaxis=dict(title='Composite Risk Score', gridcolor=COLORS['grid'], range=[0, max(df['composite_risk_score'].max() * 1.25, 0.35)]),
        yaxis=dict(title='', tickfont=dict(size=10)),
        height=max(300, top_n * 32),
        showlegend=False
    )
    return fig


def plot_mi_breakdown(risk_df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    df = risk_df.head(top_n).copy()
    df = df.sort_values('composite_risk_score', ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='MI with Outcome',
        x=df['feature'],
        y=df['mi_with_outcome'],
        marker_color=COLORS['accent'],
        opacity=0.85
    ))
    fig.add_trace(go.Bar(
        name='Max Proxy MI',
        x=df['feature'],
        y=df['max_proxy_mi'],
        marker_color=COLORS['HIGH'],
        opacity=0.85
    ))
    fig.add_trace(go.Bar(
        name='Surrogate Proxy Score',
        x=df['feature'],
        y=df['surrogate_proxy_score'],
        marker_color=COLORS['MEDIUM'],
        opacity=0.85
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text='Mutual Information Breakdown by Feature', font=dict(size=13, color=COLORS['primary'])),
        barmode='group',
        xaxis=dict(title='', tickangle=-30),
        yaxis=dict(title='Score', gridcolor=COLORS['grid']),
        legend=dict(orientation='h', y=1.12, x=0),
        height=380
    )
    return fig


def plot_fairness_radar(aggregate_metrics: dict, thresholds: dict) -> go.Figure:
    metrics = list(aggregate_metrics.keys())
    labels = [m.replace('_', ' ').replace('difference', 'diff').title() for m in metrics]
    values = [abs(aggregate_metrics[m]) for m in metrics]
    thresh_vals = [thresholds.get(m, 0.1) for m in metrics]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(200, 40, 40, 0.15)',
        line=dict(color=COLORS['HIGH'], width=2),
        name='Model Fairness Gap'
    ))

    fig.add_trace(go.Scatterpolar(
        r=thresh_vals + [thresh_vals[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(10, 150, 80, 0.08)',
        line=dict(color=COLORS['pass'], width=1.5, dash='dot'),
        name='Acceptable Threshold'
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max(values), 0.4)], gridcolor=COLORS['grid']),
            angularaxis=dict(gridcolor=COLORS['grid'])
        ),
        title=dict(text='Fairness Metric Radar', font=dict(size=13, color=COLORS['primary'])),
        legend=dict(orientation='h', y=-0.15),
        height=400
    )
    return fig


def plot_conflict_heatmap(aggregate_metrics: dict, conflict_data: list) -> go.Figure:
    metrics = list(aggregate_metrics.keys())
    short_names = [m.replace('_difference', '').replace('_', ' ').upper()[:12] for m in metrics]
    n = len(metrics)

    matrix = np.zeros((n, n))
    hover_text = [[''] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                val = abs(aggregate_metrics[metrics[i]])
                matrix[i][j] = val
                hover_text[i][j] = f"{short_names[i]}<br>Value: {val:.4f}"
            else:
                matrix[i][j] = 0.0
                hover_text[i][j] = 'No conflict'

    for conflict in conflict_data:
        m1 = conflict['metric_1']
        m2 = conflict['metric_2']
        if m1 in metrics and m2 in metrics:
            i, j = metrics.index(m1), metrics.index(m2)
            severity_score = {'HIGH': 0.9, 'MEDIUM': 0.5, 'LOW': 0.2}.get(conflict['severity'], 0.3)
            matrix[i][j] = severity_score
            matrix[j][i] = severity_score
            desc = conflict['description'][:60] + '...' if len(conflict['description']) > 60 else conflict['description']
            hover_text[i][j] = f"CONFLICT: {conflict['conflict_type']}<br>Severity: {conflict['severity']}<br>{desc}"
            hover_text[j][i] = hover_text[i][j]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=short_names,
        y=short_names,
        colorscale=[[0, '#f0f4fc'], [0.3, '#ffd580'], [0.6, '#ff8c42'], [1.0, '#c82828']],
        showscale=True,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        zmin=0, zmax=1,
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text='Fairness Metric Conflict Matrix', font=dict(size=13, color=COLORS['primary'])),
        xaxis=dict(title='', tickangle=-30, tickfont=dict(size=9)),
        yaxis=dict(title='', tickfont=dict(size=9)),
        height=380
    )
    return fig


def plot_group_comparison(group_metrics: dict, metric: str = 'positive_rate') -> go.Figure:
    groups = list(group_metrics.keys())
    values = [group_metrics[g].get(metric, 0) for g in groups]

    colors = []
    max_val = max(values) if values else 1
    for v in values:
        ratio = v / max_val if max_val > 0 else 1
        if ratio < 0.8:
            colors.append(COLORS['HIGH'])
        elif ratio < 0.9:
            colors.append(COLORS['MEDIUM'])
        else:
            colors.append(COLORS['pass'])

    fig = go.Figure(go.Bar(
        x=groups,
        y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' + metric.replace('_', ' ').title() + ': %{y:.4f}<extra></extra>'
    ))

    if values:
        max_v = max(values)
        threshold_80 = max_v * 0.80
        fig.add_hline(
            y=threshold_80, line_dash='dot', line_color=COLORS['HIGH'], line_width=1.5,
            annotation_text='80% threshold (Four-Fifths Rule)',
            annotation_position='top left',
            annotation_font_size=9
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f'Group Comparison: {metric.replace("_", " ").title()}', font=dict(size=13, color=COLORS['primary'])),
        xaxis=dict(title='Group'),
        yaxis=dict(title=metric.replace('_', ' ').title(), gridcolor=COLORS['grid']),
        height=350
    )
    return fig


def plot_tradeoff_chart(tradeoffs: dict, aggregate_metrics: dict) -> go.Figure:
    metrics = [m for m in tradeoffs if tradeoffs[m].get('accuracy_cost_pct', 0) >= 0]
    costs = [tradeoffs[m]['accuracy_cost_pct'] for m in metrics]
    gaps = [abs(aggregate_metrics.get(m, 0)) for m in metrics]
    feasible = [tradeoffs[m].get('feasible', True) for m in metrics]
    short_names = [m.replace('_difference', '').replace('_', ' ').title() for m in metrics]

    marker_colors = [COLORS['pass'] if f else COLORS['HIGH'] for f in feasible]

    fig = go.Figure(go.Scatter(
        x=costs,
        y=gaps,
        mode='markers+text',
        text=short_names,
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(
            size=14,
            color=marker_colors,
            line=dict(width=1.5, color='white'),
            symbol=['circle' if f else 'x' for f in feasible]
        ),
        hovertemplate='<b>%{text}</b><br>Accuracy Cost: %{x:.2f}%<br>Fairness Gap: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text='Accuracy vs. Fairness Trade-off', font=dict(size=13, color=COLORS['primary'])),
        xaxis=dict(title='Estimated Accuracy Cost (%)', gridcolor=COLORS['grid']),
        yaxis=dict(title='Fairness Gap (absolute)', gridcolor=COLORS['grid']),
        height=380,
        annotations=[
            dict(x=0.02, y=0.98, xref='paper', yref='paper',
                 text='Lower-left = ideal (low cost, low gap)',
                 showarrow=False, font=dict(size=9, color=COLORS['neutral']))
        ]
    )
    return fig


def plot_blackbox_impact(feature_impact_summary: dict) -> go.Figure:
    features = list(feature_impact_summary.keys())
    change_rates = [feature_impact_summary[f]['outcome_change_rate'] * 100 for f in features]
    mean_deltas = [feature_impact_summary[f]['mean_prob_delta'] for f in features]
    bias_flags = [feature_impact_summary[f]['bias_detected'] for f in features]

    colors = [COLORS['HIGH'] if b else COLORS['pass'] for b in bias_flags]

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Outcome Change Rate (%)', 'Mean Probability Shift'))

    fig.add_trace(go.Bar(
        name='Outcome Change Rate',
        x=features, y=change_rates,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in change_rates],
        textposition='outside',
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name='Mean Probability Shift',
        x=features, y=mean_deltas,
        marker_color=[COLORS['MEDIUM'] if b else COLORS['accent'] for b in bias_flags],
        text=[f"{v:.4f}" for v in mean_deltas],
        textposition='outside',
    ), row=1, col=2)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text='Black-Box Probe: Per-Feature Bias Impact', font=dict(size=13, color=COLORS['primary'])),
        showlegend=False,
        height=380
    )
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=9))
    return fig
