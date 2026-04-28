"""
app.py
Fairscope - AI-Powered Fairness Auditor for Responsible Automated Decisions
Main Streamlit application entry point.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import datetime
import time
import tempfile
import sys

sys.path.insert(0, os.path.dirname(__file__))

from modules.sensitivity_scorer import run_sensitivity_analysis, DOMAIN_LEGAL_CONTEXT
from modules.fairness_metrics import compute_all_fairness_metrics, METRIC_DESCRIPTIONS
from modules.blackbox_probe import run_systematic_probe
from modules.gemini_client import (
    generate_triple_justification,
    generate_fairness_rationale,
    generate_blackbox_interpretation,
    generate_audit_summary,
    get_api_key
)
from modules.visualizations import (
    plot_sensitivity_bar, plot_mi_breakdown, plot_fairness_radar,
    plot_conflict_heatmap, plot_group_comparison, plot_tradeoff_chart,
    plot_blackbox_impact
)

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Fairscope - AI Fairness Auditor",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --primary: #0F2850;
    --accent: #1A6FBF;
    --high: #C82828;
    --medium: #C88200;
    --low: #0A9650;
    --bg: #F5F7FA;
    --surface: #FFFFFF;
    --border: #D8E0EC;
    --text: #1A1A2E;
    --muted: #6B7280;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    color: var(--text);
}

.stApp {
    background: #F0F3F8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--primary) !important;
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #C8D8F0 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #A8C0E0 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
section[data-testid="stSidebar"] hr {
    border-color: #2A4070 !important;
}

/* Main header */
.fairscope-header {
    background: linear-gradient(135deg, #0F2850 0%, #1A4A8A 60%, #1A6FBF 100%);
    padding: 2.5rem 2rem 2rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.fairscope-header::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.fairscope-title {
    font-family: var(--mono);
    font-size: 2.2rem;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: 0.12em;
    margin: 0;
}
.fairscope-subtitle {
    font-family: var(--sans);
    font-size: 0.95rem;
    color: #A8C8F0;
    margin-top: 0.4rem;
    font-weight: 300;
}
.fairscope-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: #C8E0FF;
    font-family: var(--mono);
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    margin-right: 0.4rem;
    margin-top: 0.8rem;
    letter-spacing: 0.06em;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--primary);
    margin-bottom: 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary);
}

/* Risk badges */
.badge-high { background: #FEF0F0; color: #C82828; border: 1px solid #F8BCBC; font-family: var(--mono); font-size: 0.72rem; font-weight: 600; padding: 0.18rem 0.55rem; border-radius: 2px; letter-spacing: 0.06em; }
.badge-medium { background: #FFF8E8; color: #C88200; border: 1px solid #F5D88A; font-family: var(--mono); font-size: 0.72rem; font-weight: 600; padding: 0.18rem 0.55rem; border-radius: 2px; letter-spacing: 0.06em; }
.badge-low { background: #F0FAF4; color: #0A9650; border: 1px solid #9DE0BC; font-family: var(--mono); font-size: 0.72rem; font-weight: 600; padding: 0.18rem 0.55rem; border-radius: 2px; letter-spacing: 0.06em; }
.badge-pass { background: #F0FAF4; color: #0A9650; border: 1px solid #9DE0BC; font-family: var(--mono); font-size: 0.72rem; font-weight: 600; padding: 0.18rem 0.55rem; border-radius: 2px; }
.badge-fail { background: #FEF0F0; color: #C82828; border: 1px solid #F8BCBC; font-family: var(--mono); font-size: 0.72rem; font-weight: 600; padding: 0.18rem 0.55rem; border-radius: 2px; }

/* Justification panels */
.justification-panel {
    border-left: 3px solid var(--accent);
    background: #F5F8FD;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    border-radius: 0 4px 4px 0;
}
.justification-panel.stat { border-left-color: #1A6FBF; }
.justification-panel.moral { border-left-color: #7C3AED; }
.justification-panel.legal { border-left-color: #0A9650; }
.justification-panel.high { border-left-color: #C82828; }

.jp-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.35rem;
}
.jp-text {
    font-size: 0.88rem;
    color: #2A2A3E;
    line-height: 1.55;
}

/* Metric rows */
.metric-row {
    display: flex;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #F0F0F0;
    font-size: 0.88rem;
}
.metric-name { flex: 1; color: var(--text); }
.metric-value { font-family: var(--mono); font-size: 0.85rem; margin-right: 1rem; color: #333; }

/* Conflict box */
.conflict-box {
    background: #FEF5F0;
    border: 1px solid #F8BCBC;
    border-left: 4px solid #C82828;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}
.conflict-title {
    font-family: var(--mono);
    font-size: 0.78rem;
    font-weight: 600;
    color: #C82828;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.conflict-desc {
    font-size: 0.85rem;
    color: #5A2020;
    margin-top: 0.35rem;
    line-height: 1.5;
}

/* Section headers */
.module-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem;
}
.module-number {
    background: var(--primary);
    color: white;
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.3rem 0.65rem;
    border-radius: 2px;
    letter-spacing: 0.05em;
}
.module-title {
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--primary);
    letter-spacing: 0.04em;
}

/* Info boxes */
.info-box {
    background: #EFF4FC;
    border: 1px solid #BDD0EF;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #1A3060;
    margin: 0.75rem 0;
}
.warn-box {
    background: #FFF8E8;
    border: 1px solid #F5D88A;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #5A4000;
    margin: 0.75rem 0;
}
.success-box {
    background: #F0FAF4;
    border: 1px solid #9DE0BC;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #0A4020;
    margin: 0.75rem 0;
}
.error-box {
    background: #FEF0F0;
    border: 1px solid #F8BCBC;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #5A0000;
    margin: 0.75rem 0;
}

/* Steps indicator */
.step-indicator {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.3rem 0.7rem;
    border-radius: 2px;
}
.step-done { background: #E8F5EE; color: #0A9650; border: 1px solid #9DE0BC; }
.step-active { background: var(--primary); color: white; }
.step-pending { background: #F0F0F5; color: #9090A0; border: 1px solid #D0D0E0; }

/* Gemini text */
.gemini-output {
    background: linear-gradient(135deg, #F5F0FF 0%, #F0F5FF 100%);
    border: 1px solid #C8B8F0;
    border-left: 4px solid #7C3AED;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    line-height: 1.65;
    color: #2A1A4A;
    margin: 0.75rem 0;
    font-style: italic;
}
.gemini-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7C3AED;
    margin-bottom: 0.5rem;
    font-style: normal;
}

/* Stat number */
.stat-number {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 600;
    color: var(--primary);
    line-height: 1;
}
.stat-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
}

/* Dividers */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* Scrollable table */
.stDataFrame { border-radius: 4px; }

/* Sidebar section label */
.sidebar-section {
    font-family: var(--mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6080A0 !important;
    margin-top: 1.2rem;
    margin-bottom: 0.3rem;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------

for key in ['df', 'sensitivity_results', 'fairness_results', 'blackbox_results',
            'gemini_justifications', 'chosen_metric', 'gemini_rationale',
            'gemini_summary', 'audit_complete', 'api_key_set']:
    if key not in st.session_state:
        st.session_state[key] = None

if 'audit_complete' not in st.session_state:
    st.session_state.audit_complete = False

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p style="font-family: IBM Plex Mono, monospace; font-size: 1.4rem; font-weight: 600; color: white; letter-spacing: 0.1em; margin-bottom: 0;">FAIRSCOPE</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.75rem; color: #8090B0; margin-top: 0; margin-bottom: 1.2rem;">AI Fairness Auditor v1.0</p>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color: #2A4070; margin: 0.5rem 0 1rem;">', unsafe_allow_html=True)

    # API Key
    st.markdown('<p class="sidebar-section">Gemini API Key</p>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        label_visibility="collapsed",
        help="Get your key at aistudio.google.com"
    )
    if api_key_input:
        os.environ['GEMINI_API_KEY'] = api_key_input
        st.session_state.api_key_set = True
        st.markdown('<p style="color: #6DE0A0; font-size: 0.75rem;">API key configured</p>', unsafe_allow_html=True)
    elif get_api_key():
        st.session_state.api_key_set = True
        st.markdown('<p style="color: #6DE0A0; font-size: 0.75rem;">API key active (hardcoded)</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: #F0A060; font-size: 0.75rem;">No key set - AI justifications disabled</p>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color: #2A4070; margin: 1rem 0;">', unsafe_allow_html=True)

    # File Upload
    st.markdown('<p class="sidebar-section">Dataset</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=['csv'],
        label_visibility="collapsed",
        help="Upload your dataset as a CSV file. Use the sample generator script for a demo dataset."
    )

    use_sample = st.checkbox("Use built-in sample dataset", value=False)

    if uploaded_file is not None or use_sample:
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
        else:
            sample_path = "data/adult_income_sample.csv"
            if not os.path.exists(sample_path):
                os.makedirs("data", exist_ok=True)
                import subprocess
                subprocess.run(["python", "generate_sample_data.py"], capture_output=True)
            if os.path.exists(sample_path):
                df_raw = pd.read_csv(sample_path)
            else:
                st.error("Run generate_sample_data.py first to create the sample dataset.")
                df_raw = None

        if df_raw is not None:
            st.session_state.df = df_raw
            st.markdown(f'<p style="color: #80C0A0; font-size: 0.78rem;">{len(df_raw):,} rows x {len(df_raw.columns)} columns loaded</p>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color: #2A4070; margin: 1rem 0;">', unsafe_allow_html=True)

    # Configuration
    if st.session_state.df is not None:
        df_cols = list(st.session_state.df.columns)

        st.markdown('<p class="sidebar-section">Domain</p>', unsafe_allow_html=True)
        domain = st.selectbox(
            "Application Domain",
            ['hiring', 'lending', 'healthcare', 'insurance'],
            label_visibility="collapsed"
        )

        st.markdown('<p class="sidebar-section">Target Variable</p>', unsafe_allow_html=True)
        target_col = st.selectbox(
            "Target Column",
            df_cols,
            index=len(df_cols)-1,
            label_visibility="collapsed"
        )

        st.markdown('<p class="sidebar-section">Sensitive Attributes</p>', unsafe_allow_html=True)
        default_sensitive = [c for c in df_cols if c.lower() in
                             {'sex', 'gender', 'race', 'ethnicity', 'age', 'marital_status',
                              'religion', 'nationality', 'disability'}]
        sensitive_cols = st.multiselect(
            "Sensitive Attributes",
            [c for c in df_cols if c != target_col],
            default=default_sensitive[:2] if default_sensitive else [],
            label_visibility="collapsed"
        )

        st.markdown('<p class="sidebar-section">Privileged Group (optional)</p>', unsafe_allow_html=True)
        if sensitive_cols:
            primary_sensitive = sensitive_cols[0]
            unique_vals = ['Auto-detect'] + list(st.session_state.df[primary_sensitive].dropna().unique())
            privileged_value = st.selectbox(
                "Privileged Group Value",
                unique_vals,
                label_visibility="collapsed"
            )
            if privileged_value == 'Auto-detect':
                privileged_value = None
        else:
            privileged_value = None

        st.markdown('<hr style="border-color: #2A4070; margin: 1rem 0;">', unsafe_allow_html=True)

        run_audit = st.button(
            "RUN FULL AUDIT",
            use_container_width=True,
            type="primary"
        )
    else:
        domain = 'hiring'
        target_col = None
        sensitive_cols = []
        privileged_value = None
        run_audit = False

    st.markdown('<hr style="border-color: #2A4070; margin: 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<p style="color: #506080; font-size: 0.7rem; line-height: 1.5;">Fairscope does not store your data. All analysis runs locally. AI justifications via Gemini API. Not legal advice.</p>', unsafe_allow_html=True)


# -------------------------------------------------------------------
# MAIN CONTENT
# -------------------------------------------------------------------

# Header
st.markdown("""
<div class="fairscope-header">
    <div class="fairscope-title">FAIRSCOPE</div>
    <div class="fairscope-subtitle">AI-Powered Fairness Auditor for Responsible Automated Decisions</div>
    <div style="margin-top: 0.9rem;">
        <span class="fairscope-badge">EU AI ACT</span>
        <span class="fairscope-badge">ECOA</span>
        <span class="fairscope-badge">TITLE VII</span>
        <span class="fairscope-badge">FOUR-FIFTHS RULE</span>
        <span class="fairscope-badge">PROXY DETECTION</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# LANDING STATE
# -------------------------------------------------------------------

if st.session_state.df is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Module 01 - Proxy Detection</div>
            <p style="font-size: 0.87rem; color: #444; line-height: 1.6;">
                Detects hidden proxy features using mutual information, surrogate model scoring,
                and known proxy heuristics. Generates triple justifications (statistical, moral,
                legal) via Gemini AI for each high-risk feature.
            </p>
            <p style="font-family: IBM Plex Mono, monospace; font-size: 0.72rem; color: #1A6FBF; margin-top: 0.5rem;">
                MI SCORING + SURROGATE MODELS + INTERSECTIONAL DETECTION
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Module 02 - Fairness Conflicts</div>
            <p style="font-size: 0.87rem; color: #444; line-height: 1.6;">
                Computes five fairness metrics across sensitive groups, detects mathematical
                conflicts between definitions, visualizes accuracy-fairness trade-offs,
                and generates auditable rationale for the chosen criterion.
            </p>
            <p style="font-family: IBM Plex Mono, monospace; font-size: 0.72rem; color: #1A6FBF; margin-top: 0.5rem;">
                DPD + EOD + EQOPD + PPD + CALIBRATION
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Module 03 - Black-Box Probe</div>
            <p style="font-size: 0.87rem; color: #444; line-height: 1.6;">
                Audits models without internal access via systematic counterfactual testing.
                Changes proxy features one at a time and in combination to detect if outcomes
                shift when only demographic signals change.
            </p>
            <p style="font-family: IBM Plex Mono, monospace; font-size: 0.72rem; color: #1A6FBF; margin-top: 0.5rem;">
                COUNTERFACTUAL TESTING + MULTI-FEATURE PROBES
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Getting Started:</strong> Upload a CSV dataset using the sidebar, or check "Use built-in sample dataset" 
        (after running <code>python generate_sample_data.py</code>). Configure your domain, target variable, 
        and sensitive attributes, then click "RUN FULL AUDIT".
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top: 1rem;">
        <div class="card-title">Quick Setup Guide</div>
        <div style="font-size: 0.87rem; color: #444; line-height: 1.9;">
            1. Install dependencies: <code style="background:#F0F0F8; padding: 0.1rem 0.4rem; border-radius: 2px;">pip install -r requirements.txt</code><br>
            2. Generate sample data: <code style="background:#F0F0F8; padding: 0.1rem 0.4rem; border-radius: 2px;">python generate_sample_data.py</code><br>
            3. Add Gemini API key to <code style="background:#F0F0F8; padding: 0.1rem 0.4rem; border-radius: 2px;">.env</code> file (copy from <code>.env.example</code>)<br>
            4. Launch: <code style="background:#F0F0F8; padding: 0.1rem 0.4rem; border-radius: 2px;">streamlit run app.py</code><br>
            5. Upload data and configure settings in the sidebar
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# -------------------------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------------------------

df = st.session_state.df

with st.expander("Dataset Preview", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="stat-number">{len(df):,}</div><div class="stat-label">Rows</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-number">{len(df.columns)}</div><div class="stat-label">Columns</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-number">{df.isnull().sum().sum()}</div><div class="stat-label">Missing Values</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="stat-number">{len(df.select_dtypes(include="object").columns)}</div><div class="stat-label">Categorical Cols</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

# -------------------------------------------------------------------
# RUN AUDIT
# -------------------------------------------------------------------

if run_audit and sensitive_cols and target_col:
    if not sensitive_cols:
        st.markdown('<div class="warn-box">Please select at least one sensitive attribute in the sidebar.</div>', unsafe_allow_html=True)
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Module 1
        status_text.markdown('<div class="info-box">Running Module 1: Attribute Sensitivity Analysis...</div>', unsafe_allow_html=True)
        progress_bar.progress(10)

        sensitivity_results = run_sensitivity_analysis(df, target_col, sensitive_cols, domain)
        st.session_state.sensitivity_results = sensitivity_results
        progress_bar.progress(30)

        # Justifications generated on demand via buttons, not auto-run
        risk_df = sensitivity_results['risk_dataframe']
        st.session_state.gemini_justifications = {}
        progress_bar.progress(50)

        # Module 2
        status_text.markdown('<div class="info-box">Running Module 2: Fairness Metrics Computation...</div>', unsafe_allow_html=True)
        primary_sensitive = sensitive_cols[0]
        fairness_results = compute_all_fairness_metrics(df, target_col, primary_sensitive, privileged_value)
        st.session_state.fairness_results = fairness_results
        progress_bar.progress(70)

        # Module 3
        status_text.markdown('<div class="info-box">Running Module 3: Black-Box Counterfactual Probe...</div>', unsafe_allow_html=True)
        high_risk_features = risk_df[risk_df['risk_tier'] == 'HIGH']['feature'].tolist()[:6]
        if not high_risk_features:
            high_risk_features = risk_df.head(4)['feature'].tolist()

        blackbox_results = run_systematic_probe(
            model=fairness_results['model'],
            df=df,
            target_col=target_col,
            sensitive_cols=sensitive_cols,
            high_risk_features=high_risk_features,
            feature_cols=fairness_results['feature_cols'],
            encoders=fairness_results['encoders'],
            n_samples=40
        )
        st.session_state.blackbox_results = blackbox_results
        progress_bar.progress(85)

        st.session_state.gemini_summary = None  # generated on demand
        st.session_state.chosen_metric = None
        st.session_state.gemini_rationale = None
        st.session_state.audit_complete = True

        progress_bar.progress(100)
        status_text.markdown('<div class="success-box">Audit complete. Scroll down to review findings.</div>', unsafe_allow_html=True)

    except Exception as e:
        progress_bar.progress(0)
        status_text.markdown(f'<div class="error-box">Audit failed: {str(e)}</div>', unsafe_allow_html=True)
        st.exception(e)
        st.stop()

# -------------------------------------------------------------------
# RESULTS DISPLAY
# -------------------------------------------------------------------

if not st.session_state.audit_complete:
    st.markdown("""
    <div class="info-box" style="text-align: center; padding: 2rem;">
        Configure your dataset settings in the sidebar and click "RUN FULL AUDIT" to begin analysis.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

sensitivity_results = st.session_state.sensitivity_results
fairness_results = st.session_state.fairness_results
blackbox_results = st.session_state.blackbox_results
gemini_justifications = st.session_state.gemini_justifications or {}

# Step indicator
steps_html = """
<div class="step-indicator">
    <div class="step-item step-done">01 Proxy Detection</div>
    <div class="step-item step-done">02 Fairness Metrics</div>
    <div class="step-item step-done">03 Black-Box Probe</div>
    <div class="step-item step-done">Audit Complete</div>
</div>
"""
st.markdown(steps_html, unsafe_allow_html=True)

# Summary stats row
risk_df = sensitivity_results['risk_dataframe']
agg_metrics = fairness_results.get('aggregate_metrics', {})
n_conflicts = len(fairness_results.get('detected_conflicts', []))
bias_detected = blackbox_results.get('overall_bias_detected', False)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="card"><div class="stat-number" style="color: #C82828;">{sensitivity_results["high_risk_count"]}</div><div class="stat-label">High-Risk Proxies</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="card"><div class="stat-number" style="color: #C88200;">{sensitivity_results["medium_risk_count"]}</div><div class="stat-label">Medium-Risk Features</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="card"><div class="stat-number">{fairness_results["overall_accuracy"]*100:.1f}%</div><div class="stat-label">Model Accuracy</div></div>', unsafe_allow_html=True)
with col4:
    conflict_color = '#C82828' if n_conflicts > 0 else '#0A9650'
    st.markdown(f'<div class="card"><div class="stat-number" style="color: {conflict_color};">{n_conflicts}</div><div class="stat-label">Metric Conflicts</div></div>', unsafe_allow_html=True)
with col5:
    bias_color = '#C82828' if bias_detected else '#0A9650'
    bias_text = 'DETECTED' if bias_detected else 'CLEAR'
    st.markdown(f'<div class="card"><div class="stat-number" style="color: {bias_color}; font-size: 1.2rem; padding-top: 0.35rem;">{bias_text}</div><div class="stat-label">Black-Box Bias</div></div>', unsafe_allow_html=True)

# Executive Summary
st.markdown("""
<div class="module-header">
    <span class="module-number">SUMMARY</span>
    <span class="module-title">Executive Audit Summary</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.gemini_summary:
    st.markdown(f"""
    <div class="gemini-output">
        <div class="gemini-label">AI-Generated Summary (Gemini)</div>
        {st.session_state.gemini_summary}
    </div>
    """, unsafe_allow_html=True)
else:
    if st.button("Generate AI Executive Summary", type="secondary", key="btn_summary"):
        with st.spinner("Generating summary -- this may take a few seconds..."):
            top_proxy_names = sensitivity_results['risk_dataframe'].head(3)['feature'].tolist()
            key_findings = [
                f"{sensitivity_results['high_risk_count']} high-risk proxy features identified",
                f"Model accuracy: {fairness_results['overall_accuracy']*100:.1f}%",
                f"Detected conflicts: {len(fairness_results['detected_conflicts'])} fairness metric incompatibilities",
                f"Black-box probe bias detected: {blackbox_results['overall_bias_detected']}"
            ]
            result = generate_audit_summary(
                domain=domain,
                sensitive_attributes=sensitive_cols,
                top_proxies=top_proxy_names,
                chosen_fairness_metric='Pending user selection',
                key_findings=key_findings
            )
            st.session_state.gemini_summary = result
            st.rerun()

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ===================================================================
# MODULE 1: ATTRIBUTE SENSITIVITY
# ===================================================================

st.markdown("""
<div class="module-header">
    <span class="module-number">MODULE 01</span>
    <span class="module-title">Attribute Sensitivity Scorer</span>
</div>
""", unsafe_allow_html=True)

m1_col1, m1_col2 = st.columns([3, 2])

with m1_col1:
    fig_bar = plot_sensitivity_bar(risk_df, top_n=12)
    st.plotly_chart(fig_bar, use_container_width=True)

with m1_col2:
    fig_mi = plot_mi_breakdown(risk_df, top_n=8)
    st.plotly_chart(fig_mi, use_container_width=True)

# Risk table
st.markdown('<div class="card"><div class="card-title">Feature Risk Rankings</div>', unsafe_allow_html=True)
display_df = risk_df[['feature', 'mi_with_outcome', 'max_proxy_mi', 'surrogate_proxy_score',
                       'composite_risk_score', 'risk_tier', 'primary_proxy_target', 'is_known_proxy']].copy()
display_df.columns = ['Feature', 'MI (Outcome)', 'Max Proxy MI', 'Surrogate Score',
                      'Composite Risk', 'Risk Tier', 'Proxy For', 'Known Proxy']
st.dataframe(display_df.head(15), use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# Gemini Triple Justifications -- one button per feature, no auto-calls
st.markdown('<div class="card-title" style="margin-top: 1.5rem; padding-bottom: 0.5rem; border-bottom: 2px solid #0F2850; font-family: IBM Plex Mono, monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #0F2850;">AI Triple Justification Panel (Gemini)</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box" style="margin-bottom: 0.75rem;">Click "Analyze" on any feature to generate its statistical, moral, and legal justification via Gemini. Generate one at a time to avoid rate limits.</div>', unsafe_allow_html=True)

top_features_for_justification = risk_df[risk_df['risk_tier'].isin(['HIGH', 'MEDIUM'])].head(5)

for _, feat_row in top_features_for_justification.iterrows():
    feat = feat_row['feature']
    risk_tier = feat_row['risk_tier']
    proxy_target = feat_row['primary_proxy_target']

    with st.expander(f"{feat.upper()}  --  Risk: {risk_tier}  |  Proxy for: {proxy_target}", expanded=(risk_tier == 'HIGH')):
        jus = gemini_justifications.get(feat)

        if jus is None:
            if st.button(f"Analyze with Gemini", key=f"btn_jus_{feat}"):
                with st.spinner(f"Generating justification for {feat}..."):
                    dataset_context = f"Dataset with {len(df)} records for {domain} decisions. Target: {target_col}. Sensitive: {', '.join(sensitive_cols)}."
                    result = generate_triple_justification(
                        feature_name=feat,
                        mi_score=feat_row['mi_with_outcome'],
                        proxy_target=proxy_target,
                        domain=domain,
                        dataset_context=dataset_context
                    )
                    st.session_state.gemini_justifications[feat] = result
                    st.rerun()
        else:
            rec_action = jus.get('recommended_action', '')
            if rec_action:
                st.markdown(f'<div class="warn-box"><strong>Recommended Action:</strong> {rec_action}</div>', unsafe_allow_html=True)

            jc1, jc2, jc3 = st.columns(3)
            with jc1:
                st.markdown(f"""
                <div class="justification-panel stat">
                    <div class="jp-label">Statistical Basis</div>
                    <div class="jp-text">{jus.get('statistical', 'No data.')}</div>
                </div>
                """, unsafe_allow_html=True)
            with jc2:
                st.markdown(f"""
                <div class="justification-panel moral">
                    <div class="jp-label">Moral / Historical Context</div>
                    <div class="jp-text">{jus.get('moral_historical', 'No data.')}</div>
                </div>
                """, unsafe_allow_html=True)
            with jc3:
                st.markdown(f"""
                <div class="justification-panel legal">
                    <div class="jp-label">Legal / Regulatory Basis</div>
                    <div class="jp-text">{jus.get('legal', 'No data.')}</div>
                </div>
                """, unsafe_allow_html=True)

# Intersectional risks
intersectional = sensitivity_results.get('intersectional_risks', [])
if intersectional:
    st.markdown("""
    <div class="module-header" style="margin-top: 1.5rem;">
        <span class="module-number" style="background: #7C3AED;">BLIND SPOTS</span>
        <span class="module-title">Intersectional Risk Detection</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="info-box">The following feature pairs reveal sensitive attribute information in combination, even when each appears safe individually. This is a critical blind spot in single-feature auditing approaches.</div>', unsafe_allow_html=True)

    for risk in intersectional[:5]:
        blind_label = '<span class="badge-fail">BLIND SPOT</span>' if risk.get('blind_spot') else ''
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid #7C3AED;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-family: IBM Plex Mono, monospace; font-size: 0.88rem; font-weight: 600;">{risk['feature_1']} x {risk['feature_2']}</span>
                <span style="color: #7C3AED; font-size: 0.78rem;">proxy for {risk['sensitive_attr']}</span>
                {blind_label}
            </div>
            <div style="display: flex; gap: 2rem; font-size: 0.82rem; color: #555; font-family: IBM Plex Mono, monospace;">
                <span>MI alone (f1): {risk['mi_f1_alone']}</span>
                <span>MI alone (f2): {risk['mi_f2_alone']}</span>
                <span>MI combined: {risk['mi_combined']}</span>
                <span style="font-weight: 600; color: #7C3AED;">Synergy: +{risk['synergy_score']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ===================================================================
# MODULE 2: FAIRNESS METRICS
# ===================================================================

st.markdown("""
<div class="module-header">
    <span class="module-number">MODULE 02</span>
    <span class="module-title">Fairness Conflict Visualizer</span>
</div>
""", unsafe_allow_html=True)

thresholds = {m: METRIC_DESCRIPTIONS[m]['threshold'] for m in METRIC_DESCRIPTIONS}

m2_col1, m2_col2 = st.columns(2)

with m2_col1:
    fig_radar = plot_fairness_radar(agg_metrics, thresholds)
    st.plotly_chart(fig_radar, use_container_width=True)

with m2_col2:
    fig_heatmap = plot_conflict_heatmap(agg_metrics, fairness_results.get('detected_conflicts', []))
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Group comparison
group_metrics_data = fairness_results.get('group_metrics', {})
if group_metrics_data:
    metric_options = ['positive_rate', 'true_positive_rate', 'false_positive_rate', 'positive_predictive_value', 'accuracy']
    selected_group_metric = st.selectbox(
        "Group Comparison Metric",
        metric_options,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    fig_group = plot_group_comparison(group_metrics_data, selected_group_metric)
    st.plotly_chart(fig_group, use_container_width=True)

# Trade-off chart
tradeoffs = fairness_results.get('tradeoffs', {})
if tradeoffs:
    fig_tradeoff = plot_tradeoff_chart(tradeoffs, agg_metrics)
    st.plotly_chart(fig_tradeoff, use_container_width=True)

# Metrics detail
st.markdown('<div class="card"><div class="card-title">Fairness Metrics Detail</div>', unsafe_allow_html=True)
for metric, val in agg_metrics.items():
    thresh = thresholds.get(metric, 0.1)
    is_pass = abs(val) <= thresh
    status_badge = '<span class="badge-pass">PASS</span>' if is_pass else '<span class="badge-fail">FAIL</span>'
    desc = METRIC_DESCRIPTIONS.get(metric, {})
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-name">
            <strong>{desc.get('name', metric)}</strong>
            <span style="color: #9090A0; font-size: 0.78rem; margin-left: 0.5rem;">({desc.get('short', '')})</span>
            <div style="font-size: 0.78rem; color: #808090; margin-top: 0.2rem;">{desc.get('definition', '')}</div>
        </div>
        <div class="metric-value">{val:.4f}</div>
        {status_badge}
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Conflict report
conflicts = fairness_results.get('detected_conflicts', [])
if conflicts:
    st.markdown('<div class="card-title" style="margin-top: 1.5rem; padding-bottom: 0.5rem; border-bottom: 2px solid #C82828; font-family: IBM Plex Mono, monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #C82828;">Detected Fairness Conflicts</div>', unsafe_allow_html=True)
    for c in conflicts:
        st.markdown(f"""
        <div class="conflict-box">
            <div class="conflict-title">[{c['conflict_type']}] {c['metric_1'].replace('_',' ').upper()} vs {c['metric_2'].replace('_',' ').upper()}</div>
            <div class="conflict-desc">{c['description']}</div>
            <div style="margin-top: 0.4rem; font-size: 0.78rem; font-family: IBM Plex Mono, monospace; color: #C82828;">
                Severity: {c['severity']} &nbsp;|&nbsp; M1 value: {c['metric_1_value']:.4f} &nbsp;|&nbsp; M2 value: {c['metric_2_value']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Fairness criterion selection + rationale generation
st.markdown('<div class="card" style="margin-top: 1.5rem;"><div class="card-title">Select Fairness Criterion and Generate Audit Rationale</div>', unsafe_allow_html=True)
metric_display_names = {m: METRIC_DESCRIPTIONS[m]['name'] for m in agg_metrics}
chosen_metric_name = st.selectbox(
    "Choose the fairness definition your organization will prioritize:",
    list(metric_display_names.values())
)
chosen_metric_key = [k for k, v in metric_display_names.items() if v == chosen_metric_name][0]

user_justification = st.text_area(
    "Optional: Add your organization's reasoning for this choice (will be included in audit record):",
    placeholder="e.g., Our legal team determined that Equal Opportunity is most aligned with ECOA requirements for our lending context...",
    height=80
)

if st.button("Generate Audit Rationale via Gemini", type="secondary"):
    with st.spinner("Generating rationale..."):
        rationale = generate_fairness_rationale(
            chosen_metric=chosen_metric_name,
            metric_values=agg_metrics,
            domain=domain if 'domain' in dir() else 'hiring',
            sensitive_attribute=fairness_results.get('sensitive_col', 'sensitive attribute'),
            accuracy_cost=tradeoffs.get(chosen_metric_key, {}).get('accuracy_cost_pct', 0.0)
        )
        if user_justification:
            rationale = rationale + "\n\nOrganization Note: " + user_justification
        st.session_state.chosen_metric = chosen_metric_key
        st.session_state.gemini_rationale = rationale

if st.session_state.gemini_rationale:
    st.markdown(f"""
    <div class="gemini-output">
        <div class="gemini-label">AI-Generated Audit Rationale (Gemini) -- Timestamp: {datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}</div>
        {st.session_state.gemini_rationale}
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ===================================================================
# MODULE 3: BLACK-BOX PROBE
# ===================================================================

st.markdown("""
<div class="module-header">
    <span class="module-number">MODULE 03</span>
    <span class="module-title">Black-Box Counterfactual Probe</span>
</div>
""", unsafe_allow_html=True)

if blackbox_results:
    bias_overall = blackbox_results.get('overall_bias_detected', False)
    if bias_overall:
        st.markdown('<div class="error-box"><strong>BIAS DETECTED:</strong> Counterfactual testing found that changing proxy features significantly alters model outcomes for otherwise identical records. This indicates the model may be discriminating based on demographic signals embedded in non-sensitive features.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box"><strong>No Significant Bias Detected:</strong> Counterfactual tests did not reveal systematic outcome changes attributable to proxy feature manipulation. Continue standard monitoring.</div>', unsafe_allow_html=True)

    bb_c1, bb_c2, bb_c3 = st.columns(3)
    with bb_c1:
        st.markdown(f'<div class="card"><div class="stat-number">{blackbox_results["total_tests_run"]:,}</div><div class="stat-label">Counterfactual Tests</div></div>', unsafe_allow_html=True)
    with bb_c2:
        rate_color = '#C82828' if blackbox_results['combined_bias_rate'] > 0.10 else '#0A9650'
        st.markdown(f'<div class="card"><div class="stat-number" style="color: {rate_color};">{blackbox_results["combined_bias_rate"]*100:.1f}%</div><div class="stat-label">Combined Proxy Bias Rate</div></div>', unsafe_allow_html=True)
    with bb_c3:
        delta_color = '#C82828' if blackbox_results['combined_mean_prob_delta'] > 0.05 else '#0A9650'
        st.markdown(f'<div class="card"><div class="stat-number" style="color: {delta_color};">{blackbox_results["combined_mean_prob_delta"]:.4f}</div><div class="stat-label">Mean Probability Shift</div></div>', unsafe_allow_html=True)

    feat_summary = blackbox_results.get('feature_impact_summary', {})
    if feat_summary:
        fig_bb = plot_blackbox_impact(feat_summary)
        st.plotly_chart(fig_bb, use_container_width=True)

        st.markdown('<div class="card"><div class="card-title">Per-Feature Bias Breakdown</div>', unsafe_allow_html=True)
        for feat, summary in feat_summary.items():
            bias_flag = summary.get('bias_detected', False)
            badge = '<span class="badge-fail">BIAS DETECTED</span>' if bias_flag else '<span class="badge-pass">CLEAR</span>'
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-name"><strong>{feat}</strong></div>
                <div class="metric-value" style="font-size: 0.78rem;">
                    Change rate: {summary['outcome_change_rate']*100:.1f}%
                    &nbsp;|&nbsp; Mean delta: {summary['mean_prob_delta']:.4f}
                    &nbsp;|&nbsp; Max delta: {summary['max_prob_delta']:.4f}
                </div>
                {badge}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Combined probe results
    combined_probes = blackbox_results.get('combined_probe_results', [])
    if combined_probes:
        st.markdown('<div class="card" style="margin-top: 1rem;"><div class="card-title">Sample Multi-Feature Counterfactual Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box" style="margin-bottom: 0.75rem;">These tests simultaneously changed multiple proxy features to simulate a complete demographic shift, revealing compounded bias effects.</div>', unsafe_allow_html=True)

        outcome_changed_count = sum(1 for r in combined_probes if r['outcome_changed'])

        for i, result in enumerate(combined_probes[:5]):
            changed_str = ", ".join([f"{c['feature']}: {c['original']} -> {c['changed']}" for c in result['feature_changes']])
            outcome_badge = '<span class="badge-fail">OUTCOME CHANGED</span>' if result['outcome_changed'] else '<span class="badge-pass">STABLE</span>'
            delta_str = f"{result['probability_delta']:+.4f}"
            delta_color = '#C82828' if abs(result['probability_delta']) > 0.05 else '#0A9650'
            st.markdown(f"""
            <div style="padding: 0.5rem 0; border-bottom: 1px solid #F0F0F0; font-size: 0.83rem;">
                <span style="font-family: IBM Plex Mono, monospace; color: #555;">Test {i+1}:</span>
                <span style="color: #333; margin-left: 0.5rem;">{changed_str}</span>
                &nbsp;&nbsp;
                <span style="font-family: IBM Plex Mono, monospace; color: {delta_color};">{delta_str} prob delta</span>
                &nbsp;&nbsp;{outcome_badge}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f'<div style="margin-top: 0.75rem; font-size: 0.85rem; color: #555;"><strong>{outcome_changed_count} of {len(combined_probes)}</strong> combined probe tests resulted in outcome changes.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Gemini interpretation
    if combined_probes and any(r['outcome_changed'] for r in combined_probes[:3]):
        if st.button("Generate Gemini Interpretation of Counterfactual Findings", type="secondary"):
            with st.spinner("Interpreting counterfactual results..."):
                sample_result = next((r for r in combined_probes if r['outcome_changed']), combined_probes[0])
                orig_label = 'positive outcome' if sample_result['original_prediction'] == 1 else 'negative outcome'
                new_label = 'positive outcome' if sample_result['counterfactual_prediction'] == 1 else 'negative outcome'
                interpretation = generate_blackbox_interpretation(
                    feature_changes=sample_result['feature_changes'],
                    original_outcome=orig_label,
                    new_outcome=new_label,
                    domain=domain if 'domain' in dir() else 'hiring'
                )
                st.markdown(f"""
                <div class="gemini-output">
                    <div class="gemini-label">AI Counterfactual Interpretation (Gemini)</div>
                    {interpretation}
                </div>
                """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ===================================================================
# AUDIT REPORT DOWNLOAD
# ===================================================================

st.markdown("""
<div class="module-header">
    <span class="module-number">EXPORT</span>
    <span class="module-title">Audit Report Export</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">Generate PDF Audit Report</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">The audit report includes all findings, metric values, conflict analysis, justification text, and timestamps in a format suitable for regulatory documentation and compliance review.</div>', unsafe_allow_html=True)

if st.button("Generate and Download PDF Audit Report", type="primary"):
    with st.spinner("Generating PDF report..."):
        try:
            from modules.report_generator import generate_audit_report
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                report_path = tmp.name

            generate_audit_report(
                domain=domain if 'domain' in dir() else 'hiring',
                sensitive_cols=sensitive_cols if 'sensitive_cols' in dir() else [],
                target_col=target_col if 'target_col' in dir() else 'target',
                sensitivity_results=sensitivity_results,
                fairness_results=fairness_results,
                blackbox_results=blackbox_results,
                chosen_metric=st.session_state.chosen_metric or 'Not selected',
                gemini_rationale=st.session_state.gemini_rationale or '',
                gemini_summary=st.session_state.gemini_summary or '',
                output_path=report_path
            )

            with open(report_path, 'rb') as f:
                pdf_bytes = f.read()

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="Download Fairscope Audit Report (PDF)",
                data=pdf_bytes,
                file_name=f"fairscope_audit_{timestamp}.pdf",
                mime='application/pdf'
            )
            st.markdown('<div class="success-box">PDF generated. Click above to download.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-box">PDF generation failed: {str(e)}</div>', unsafe_allow_html=True)

# JSON export
if st.button("Export Raw Findings as JSON", type="secondary"):
    export_data = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'domain': domain if 'domain' in dir() else 'unknown',
        'target_col': target_col if 'target_col' in dir() else 'unknown',
        'sensitive_cols': sensitive_cols if 'sensitive_cols' in dir() else [],
        'sensitivity_summary': {
            'high_risk_count': sensitivity_results['high_risk_count'],
            'medium_risk_count': sensitivity_results['medium_risk_count'],
            'low_risk_count': sensitivity_results['low_risk_count'],
            'top_features': sensitivity_results['risk_dataframe'].head(10).to_dict('records')
        },
        'fairness_metrics': {
            'aggregate': fairness_results['aggregate_metrics'],
            'overall_accuracy': fairness_results['overall_accuracy'],
            'conflicts_detected': len(fairness_results['detected_conflicts']),
        },
        'blackbox_results': {
            'overall_bias_detected': blackbox_results.get('overall_bias_detected'),
            'combined_bias_rate': blackbox_results.get('combined_bias_rate'),
            'total_tests': blackbox_results.get('total_tests_run')
        },
        'chosen_metric': st.session_state.chosen_metric,
        'gemini_rationale': st.session_state.gemini_rationale
    }

    st.download_button(
        label="Download JSON Export",
        data=json.dumps(export_data, indent=2, default=str),
        file_name=f"fairscope_findings_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime='application/json'
    )

st.markdown('</div>', unsafe_allow_html=True)

# Legal disclaimer
st.markdown("""
<div style="margin-top: 2rem; padding: 1rem; background: #F8F8FA; border: 1px solid #E0E0E8; border-radius: 4px; font-size: 0.78rem; color: #808090; line-height: 1.6;">
    <strong>Disclaimer:</strong> Fairscope is an automated analysis tool and does not constitute legal advice.
    AI-generated justifications (Gemini) may not reflect current law and should be reviewed by qualified legal counsel.
    Fairness metrics are statistical measures and do not guarantee regulatory compliance.
    Organizations remain solely responsible for their AI systems' compliance with applicable law.
    No personally identifiable data is transmitted or stored by Fairscope beyond your local session.
</div>
""", unsafe_allow_html=True)