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
# CUSTOM CSS  -- palette: #6E1A37  #AE2448  #72BAA9  #D5E7B5
# -------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap');
            
:root {
    --p1: #6E1A37;
    --p2: #AE2448;
    --p3: #72BAA9;
    --p4: #D5E7B5;
    --white: #FFFFFF;
    --soft: #F0E9E4;
}

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background-color: var(--p4) !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background-color: var(--p1) !important;
}
[data-testid="stSidebar"] * { color: var(--soft) !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background-color: #8B2A4A !important;
    color: var(--soft) !important;
    border-color: var(--p2) !important;
}

/* File Uploader - Make text clearly visible */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div
[data-testid="stFileUploader"] button {
    color: #6E1A37 !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] button {
    background-color: #72BAA9 !important;
    color: #6E1A37 !important;
}

/* ---- Header ---- */
.fs-header {
    background: var(--p1);
    padding: 2.4rem 2.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}
.fs-title {
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--soft);
    letter-spacing: 0.04em;
    font-family: 'Roboto Mono', monospace;
}
.fs-sub {
    font-size: 1.08rem;
    color: var(--p3);
    margin-top: 0.45rem;
}
.fs-badge {
    display: inline-block;
    background: var(--p2);
    color: var(--soft) !important;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.73rem;
    padding: 0.32rem 0.85rem;
    border-radius: 999px;
    margin-right: 0.5rem;
    margin-top: 0.6rem;
    text-decoration: none;
}
.fs-badge:hover { 
    background: var(--p3); 
    color: var(--p1) !important; 
}

/* ---- White surface card (for charts and content blocks) ---- */
.surface {
    background: var(--white);
    border: 1px solid var(--p3);
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
}

/* ---- Standard card (p4 background) ---- */
.card {
    background: var(--white);
    border: 1px solid var(--p3);
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 8px rgba(110,26,55,0.07);
}
.card-title {
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--p1);
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid var(--p3);
}

/* ---- Module headers ---- */
.mod-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2.2rem 0 1.2rem;
}
.mod-num {
    background: var(--p1);
    color: var(--soft);
    font-family: 'Roboto Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.42rem 0.95rem;
    border-radius: 6px;
    letter-spacing: 0.03em;
}
.mod-title {
    font-size: 1.38rem;
    font-weight: 500;
    color: var(--p1);
}

/* ---- KPI stat cards ---- */
.kpi-card {
    background: var(--white);
    border: 1px solid var(--p3);
    border-radius: 10px;
    padding: 1.3rem 1.5rem 1.1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(110,26,55,0.06);
}
.stat-num {
    font-family: 'Roboto Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: var(--p1);
    line-height: 1.1;
}
.stat-label {
    font-size: 0.73rem;
    color: var(--p2);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.35rem;
}

/* ---- Step bar ---- */
.step-bar {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
}
.step-done {
    background: var(--p3);
    color: var(--p1);
    border: 1px solid var(--p1);
    font-family: 'Roboto Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ---- Risk / status badges ---- */
.b-high, .b-fail {
    background: var(--p2);
    color: var(--soft);
    font-family: 'Roboto Mono', monospace;
    font-size: 0.74rem;
    font-weight: 500;
    padding: 0.28rem 0.65rem;
    border-radius: 5px;
}
.b-medium {
    background: var(--p1);
    color: var(--soft);
    font-family: 'Roboto Mono', monospace;
    font-size: 0.74rem;
    font-weight: 500;
    padding: 0.28rem 0.65rem;
    border-radius: 5px;
}
.b-low, .b-pass {
    background: var(--p3);
    color: var(--p1);
    font-family: 'Roboto Mono', monospace;
    font-size: 0.74rem;
    font-weight: 500;
    padding: 0.28rem 0.65rem;
    border-radius: 5px;
}

/* ---- Info / status boxes ---- */
.box-info {
    background: var(--soft);
    border: 1px solid var(--p3);
    border-left: 4px solid var(--p3);
    border-radius: 8px;
    padding: 1rem 1.3rem;
    font-size: 0.93rem;
    color: var(--p1);
    margin: 0.85rem 0;
    line-height: 1.65;
}
.box-warn {
    background: var(--soft);
    border: 1px solid var(--p1);
    border-left: 4px solid var(--p1);
    border-radius: 8px;
    padding: 1rem 1.3rem;
    font-size: 0.93rem;
    color: var(--p1);
    margin: 0.85rem 0;
}
.box-success {
    background: var(--p3);
    border-left: 4px solid var(--p1);
    border-radius: 8px;
    padding: 1rem 1.3rem;
    font-size: 0.93rem;
    color: var(--p1);
    margin: 0.85rem 0;
}
.box-error {
    background: var(--p2);
    border-left: 4px solid var(--p1);
    border-radius: 8px;
    padding: 1rem 1.3rem;
    font-size: 0.93rem;
    color: var(--soft);
    margin: 0.85rem 0;
}

/* ---- Gemini output ---- */
.gemini-out {
    background: var(--white);
    border: 1px solid var(--p1);
    border-left: 4px solid var(--p1);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    font-size: 0.91rem;
    line-height: 1.7;
    color: var(--p1);
    margin: 0.9rem 0;
}
.gemini-label {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--p2);
    margin-bottom: 0.5rem;
}

/* ---- Justification panels ---- */
.jp {
    background: var(--soft);
    border-left: 4px solid var(--p3);
    padding: 1.1rem 1.25rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
}
.jp.stat  { border-left-color: var(--p3); }
.jp.moral { border-left-color: var(--p1); }
.jp.legal { border-left-color: var(--p2); }
.jp-label {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--p2);
    margin-bottom: 0.4rem;
}
.jp-text { font-size: 0.9rem; line-height: 1.62; color: var(--p1); }

/* ---- Conflict box ---- */
.conflict-box {
    background: var(--white);
    border: 1px solid var(--p2);
    border-left: 4px solid var(--p2);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0;
}
.conflict-title {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--p2);
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.conflict-desc { font-size: 0.87rem; color: var(--p1); margin-top: 0.3rem; line-height: 1.55; }

/* ---- Metric row ---- */
.metric-row {
    display: flex;
    align-items: flex-start;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--p4);
    gap: 1rem;
}
.metric-name { flex: 1; color: var(--p1); }
.metric-val {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.9rem;
    color: var(--p1);
    min-width: 72px;
    text-align: right;
    font-weight: 500;
}

/* ---- Intersectional card ---- */
.isect-card {
    background: var(--white);
    border: 1px solid var(--p3);
    border-left: 4px solid var(--p1);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
}

/* ---- Metric reference links ---- */
.metric-ref {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.74rem;
    color: var(--p2);
    text-decoration: none;
    border-bottom: 1px dotted var(--p2);
    font-weight: 500;
}
.metric-ref:hover { color: var(--p1); border-bottom-color: var(--p1); }

/* ---- Counterfactual test row ---- */
.cf-row {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid var(--p4);
    font-size: 0.84rem;
    color: var(--p1);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.cf-idx {
    font-family: 'Roboto Mono', monospace;
    color: var(--p2);
    font-size: 0.78rem;
    min-width: 50px;
}
.cf-delta {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ---- Sidebar section label ---- */
.sb-sec {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--p3);
    margin-top: 1rem;
    margin-bottom: 0.2rem;
}

/* ---- Divider ---- */
.divider { border: none; border-top: 2px solid var(--p3); margin: 2.4rem 0; }

/* ---- Disclaimer ---- */
.disclaimer {
    margin-top: 2rem;
    padding: 1.1rem 1.4rem;
    background: var(--white);
    border: 1px solid var(--p3);
    border-radius: 8px;
    font-size: 0.79rem;
    color: var(--p1);
    line-height: 1.65;
}
            
/* RUN FULL AUDIT Button - Strong default color */
.stButton > button[kind="primary"] {
    background-color: #AE2448 !important;
    color: #F0E9E4 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.85rem 1.6rem !important;
    border-radius: 8px !important;
    width: 100% !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #8B2A4A !important;
    color: #F0E9E4 !important;
}
            
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# RESEARCH LINKS
# -------------------------------------------------------------------

BADGE_LINKS = {
    "EU AI ACT":        "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
    "ECOA":             "https://www.consumerfinance.gov/compliance/supervisory-guidance/equal-credit-opportunity-act-ecoa/",
    "TITLE VII":        "https://www.eeoc.gov/statutes/title-vii-civil-rights-act-1964",
    "FOUR-FIFTHS RULE": "https://www.govinfo.gov/content/pkg/CFR-2021-title29-vol4/pdf/CFR-2021-title29-vol4-part1607.pdf",
    "PROXY DETECTION":  "https://arxiv.org/abs/1610.02413",
}

METRIC_LINKS = {
    "demographic_parity_difference": {
        "short": "DPD", "full": "Demographic Parity Difference",
        "paper": "https://fairmlbook.org/",
        "paper_label": "Barocas et al., Fairness and Machine Learning (2023)",
    },
    "equalized_odds_difference": {
        "short": "EOD", "full": "Equalized Odds Difference",
        "paper": "https://arxiv.org/abs/1610.02158",
        "paper_label": "Hardt et al., Equality of Opportunity in Supervised Learning (2016)",
    },
    "equal_opportunity_difference": {
        "short": "EQOPD", "full": "Equal Opportunity Difference",
        "paper": "https://arxiv.org/abs/1610.02158",
        "paper_label": "Hardt et al., Equality of Opportunity in Supervised Learning (2016)",
    },
    "predictive_parity_difference": {
        "short": "PPD", "full": "Predictive Parity Difference",
        "paper": "https://arxiv.org/abs/1609.05807",
        "paper_label": "Chouldechova, Fair Prediction with Disparate Impact (2017)",
    },
    "calibration_difference": {
        "short": "CAL", "full": "Calibration Difference",
        "paper": "https://arxiv.org/abs/1609.05807",
        "paper_label": "Chouldechova, Fair Prediction with Disparate Impact (2017)",
    },
}

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
    st.markdown('<p style="font-family: Roboto Mono, monospace; font-size: 1.3rem; font-weight: 600; color: #F0E9E4; letter-spacing: 0.08em; margin-bottom: 0;">FAIRSCOPE</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.75rem; color: #72BAA9; margin-top: 0.1rem; margin-bottom: 1rem;">AI Fairness Auditor v1.0</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color: #AE2448; margin: 0.4rem 0 1rem;">', unsafe_allow_html=True)

    st.markdown('<p class="sb-sec">Gemini API Key</p>', unsafe_allow_html=True)
    api_key_input = st.text_input("Gemini API Key", type="password", placeholder="AIza...", label_visibility="collapsed", help="Get your key at aistudio.google.com")
    if api_key_input:
        os.environ['GEMINI_API_KEY'] = api_key_input
        st.session_state.api_key_set = True
        st.markdown('<p style="color: #72BAA9; font-size: 0.75rem;">API key configured</p>', unsafe_allow_html=True)
    elif get_api_key():
        st.session_state.api_key_set = True
        st.markdown('<p style="color: #72BAA9; font-size: 0.75rem;">API key active</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: #AE2448; font-size: 0.75rem;">No key -- AI justifications disabled</p>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color: #AE2448; margin: 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<p class="sb-sec">Dataset</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV", 
        type=['csv'], 
        label_visibility="collapsed",
        help="Supported format: CSV"
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
            df_raw = pd.read_csv(sample_path) if os.path.exists(sample_path) else None
        if df_raw is not None:
            st.session_state.df = df_raw
            st.markdown(f'<p style="color: #72BAA9; font-size: 0.78rem;">{len(df_raw):,} rows x {len(df_raw.columns)} cols loaded</p>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color: #AE2448; margin: 1rem 0;">', unsafe_allow_html=True)

    if st.session_state.df is not None:
        df_cols = list(st.session_state.df.columns)
        st.markdown('<p class="sb-sec">Domain</p>', unsafe_allow_html=True)
        domain = st.selectbox("Domain", ['hiring', 'lending', 'healthcare', 'insurance'], label_visibility="collapsed")
        st.markdown('<p class="sb-sec">Target Variable</p>', unsafe_allow_html=True)
        target_col = st.selectbox("Target", df_cols, index=len(df_cols)-1, label_visibility="collapsed")
        st.markdown('<p class="sb-sec">Sensitive Attributes</p>', unsafe_allow_html=True)
        default_sensitive = [c for c in df_cols if c.lower() in {'sex','gender','race','ethnicity','age','marital_status','religion','nationality','disability'}]
        sensitive_cols = st.multiselect("Sensitive Attributes", [c for c in df_cols if c != target_col], default=default_sensitive[:2] if default_sensitive else [], label_visibility="collapsed")
        st.markdown('<p class="sb-sec">Privileged Group (optional)</p>', unsafe_allow_html=True)
        if sensitive_cols:
            primary_sensitive = sensitive_cols[0]
            unique_vals = ['Auto-detect'] + list(st.session_state.df[primary_sensitive].dropna().unique())
            privileged_value = st.selectbox("Privileged Group", unique_vals, label_visibility="collapsed")
            if privileged_value == 'Auto-detect':
                privileged_value = None
        else:
            privileged_value = None
        st.markdown('<hr style="border-color: #AE2448; margin: 1rem 0;">', unsafe_allow_html=True)
        run_audit = st.button("RUN FULL AUDIT", use_container_width=True, type="primary")
    else:
        domain = 'hiring'; target_col = None; sensitive_cols = []; privileged_value = None; run_audit = False

    st.markdown('<hr style="border-color: #AE2448; margin: 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<p style="color: #72BAA9; font-size: 0.7rem; line-height: 1.55;">Fairscope does not store your data. All analysis runs locally. AI justifications via Gemini API. Not legal advice.</p>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------

badges_html = "".join([f'<a class="fs-badge" href="{url}" target="_blank" rel="noopener">{label}</a>' for label, url in BADGE_LINKS.items()])

st.markdown(f"""
<div class="fs-header">
    <div class="fs-title">FAIRSCOPE</div>
    <div class="fs-sub" style="color: #F0E9E4 !important;>AI-Powered Fairness Auditor for Responsible Automated Decisions</div>
    <div style="margin-top: 0.6rem;">{badges_html}</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# LANDING STATE
# -------------------------------------------------------------------

if st.session_state.df is None:
    c1, c2, c3 = st.columns(3)
    modules = [
        ("Module 01 -- Proxy Detection",
         "Detects hidden proxy features using mutual information, surrogate model scoring, and known proxy heuristics. Generates triple justifications (statistical, moral, legal) via Gemini for each high-risk feature.",
         "MI SCORING + SURROGATE MODELS + INTERSECTIONAL DETECTION"),
        ("Module 02 -- Fairness Conflicts",
         "Computes five fairness metrics across sensitive groups, detects mathematical conflicts between definitions, visualizes accuracy-fairness trade-offs, and generates auditable rationale for the chosen criterion.",
         "DPD + EOD + EQOPD + PPD + CALIBRATION"),
        ("Module 03 -- Black-Box Probe",
         "Audits models without internal access via systematic counterfactual testing. Changes proxy features one at a time and in combination to detect if outcomes shift when only demographic signals change.",
         "COUNTERFACTUAL TESTING + MULTI-FEATURE PROBES"),
    ]
    for col, (title, body, tag) in zip([c1, c2, c3], modules):
        with col:
            st.markdown(f'<div class="card"><div class="card-title">{title}</div><p style="font-size:0.92rem; color:#3C2020; line-height:1.65; margin-bottom:0.8rem;">{body}</p><p style="font-family: Roboto Mono, monospace; font-size:0.71rem; color:#72BAA9; margin:0;">{tag}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="box-info"><strong>Getting Started:</strong><br><br>1. Upload your CSV dataset using the sidebar, or check "Use built-in sample dataset"<br>2. Enter your Gemini API key<br>3. Select Domain, Target Variable, and Sensitive Attributes<br>4. Click <strong>RUN FULL AUDIT</strong></div>', unsafe_allow_html=True)
    st.stop()

# -------------------------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------------------------

df = st.session_state.df

with st.expander("Dataset Preview", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip([c1,c2,c3,c4],
        [f"{len(df):,}", len(df.columns), df.isnull().sum().sum(), len(df.select_dtypes(include='object').columns)],
        ["Rows", "Columns", "Missing Values", "Categorical Cols"]):
        col.markdown(f'<div style="text-align:center; padding:0.5rem 0;"><div class="stat-num">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

# -------------------------------------------------------------------
# RUN AUDIT
# -------------------------------------------------------------------

if run_audit and sensitive_cols and target_col:
    progress_bar = st.progress(0)
    status_text  = st.empty()
    try:
        status_text.markdown('<div class="box-info">Running Module 1: Attribute Sensitivity Analysis...</div>', unsafe_allow_html=True)
        progress_bar.progress(10)
        sensitivity_results = run_sensitivity_analysis(df, target_col, sensitive_cols, domain)
        st.session_state.sensitivity_results = sensitivity_results
        progress_bar.progress(30)
        st.session_state.gemini_justifications = {}
        progress_bar.progress(50)

        status_text.markdown('<div class="box-info">Running Module 2: Fairness Metrics Computation...</div>', unsafe_allow_html=True)
        fairness_results = compute_all_fairness_metrics(df, target_col, sensitive_cols[0], privileged_value)
        st.session_state.fairness_results = fairness_results
        progress_bar.progress(70)

        status_text.markdown('<div class="box-info">Running Module 3: Black-Box Counterfactual Probe...</div>', unsafe_allow_html=True)
        rdf = sensitivity_results['risk_dataframe']
        hrf = rdf[rdf['risk_tier'] == 'HIGH']['feature'].tolist()[:6] or rdf.head(4)['feature'].tolist()
        blackbox_results = run_systematic_probe(
            model=fairness_results['model'], df=df, target_col=target_col,
            sensitive_cols=sensitive_cols, high_risk_features=hrf,
            feature_cols=fairness_results['feature_cols'], encoders=fairness_results['encoders'], n_samples=40
        )
        st.session_state.blackbox_results = blackbox_results
        progress_bar.progress(85)
        st.session_state.gemini_summary   = None
        st.session_state.chosen_metric    = None
        st.session_state.gemini_rationale = None
        st.session_state.audit_complete   = True
        progress_bar.progress(100)
        status_text.markdown('<div class="box-success">Audit complete. Scroll down to review findings.</div>', unsafe_allow_html=True)
    except Exception as e:
        progress_bar.progress(0)
        status_text.markdown(f'<div class="box-error">Audit failed: {str(e)}</div>', unsafe_allow_html=True)
        st.exception(e); st.stop()

# -------------------------------------------------------------------
# RESULTS GUARD
# -------------------------------------------------------------------

if not st.session_state.audit_complete:
    st.markdown('<div class="box-info" style="text-align:center; padding:2rem;">Configure settings in the sidebar and click RUN FULL AUDIT.</div>', unsafe_allow_html=True)
    st.stop()

sensitivity_results   = st.session_state.sensitivity_results
fairness_results      = st.session_state.fairness_results
blackbox_results      = st.session_state.blackbox_results
gemini_justifications = st.session_state.gemini_justifications or {}
risk_df               = sensitivity_results['risk_dataframe']
agg_metrics           = fairness_results.get('aggregate_metrics', {})
n_conflicts           = len(fairness_results.get('detected_conflicts', []))
bias_detected         = blackbox_results.get('overall_bias_detected', False)
tradeoffs             = fairness_results.get('tradeoffs', {})
thresholds            = {m: METRIC_DESCRIPTIONS[m]['threshold'] for m in METRIC_DESCRIPTIONS}

# Step bar
st.markdown('<div class="step-bar"><div class="step-done">01 Proxy Detection</div><div class="step-done">02 Fairness Metrics</div><div class="step-done">03 Black-Box Probe</div><div class="step-done">Audit Complete</div></div>', unsafe_allow_html=True)

# ---- KPI ROW ----
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (sensitivity_results["high_risk_count"],          "#AE2448", "High-Risk Proxies"),
    (sensitivity_results["medium_risk_count"],        "#6E1A37", "Medium-Risk Features"),
    (f'{fairness_results["overall_accuracy"]*100:.1f}%', "#6E1A37", "Model Accuracy"),
    (n_conflicts, "#AE2448" if n_conflicts > 0 else "#72BAA9", "Metric Conflicts"),
    ("DETECTED" if bias_detected else "CLEAR", "#AE2448" if bias_detected else "#72BAA9", "Black-Box Bias"),
]
for col, (val, color, label) in zip([k1,k2,k3,k4,k5], kpis):
    col.markdown(f'<div class="kpi-card"><div class="stat-num" style="color:{color};">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- SUMMARY ----
st.markdown('<div class="mod-header"><span class="mod-num">SUMMARY</span><span class="mod-title">Executive Audit Summary</span></div>', unsafe_allow_html=True)
st.markdown('<div class="surface">', unsafe_allow_html=True)
if st.session_state.gemini_summary:
    st.markdown(f'<div class="gemini-out"><div class="gemini-label">AI-Generated Summary (Gemini)</div>{st.session_state.gemini_summary}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="box-info">Click the button below to generate an AI executive summary of all findings.</div>', unsafe_allow_html=True)
    if st.button("Generate AI Executive Summary", type="secondary", key="btn_summary"):
        with st.spinner("Generating summary..."):
            result = generate_audit_summary(
                domain=domain,
                sensitive_attributes=sensitive_cols,
                top_proxies=risk_df.head(3)['feature'].tolist(),
                chosen_fairness_metric='Pending user selection',
                key_findings=[
                    f"{sensitivity_results['high_risk_count']} high-risk proxy features identified",
                    f"Model accuracy: {fairness_results['overall_accuracy']*100:.1f}%",
                    f"Detected conflicts: {n_conflicts} fairness metric incompatibilities",
                    f"Black-box probe bias detected: {blackbox_results['overall_bias_detected']}"
                ]
            )
            st.session_state.gemini_summary = result
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ===================================================================
# MODULE 01 -- ATTRIBUTE SENSITIVITY SCORER
# ===================================================================

st.markdown('<div class="mod-header"><span class="mod-num">MODULE 01</span><span class="mod-title">Attribute Sensitivity Scorer</span></div>', unsafe_allow_html=True)

# Charts in white surface cards
ch1, ch2 = st.columns([3, 2])
with ch1:
    st.markdown('<div class="surface">', unsafe_allow_html=True)
    st.plotly_chart(plot_sensitivity_bar(risk_df, top_n=12), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with ch2:
    st.markdown('<div class="surface">', unsafe_allow_html=True)
    st.plotly_chart(plot_mi_breakdown(risk_df, top_n=8), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Risk table
st.markdown('<div class="card"><div class="card-title">Feature Risk Rankings</div>', unsafe_allow_html=True)
disp = risk_df[['feature','mi_with_outcome','max_proxy_mi','surrogate_proxy_score',
                 'composite_risk_score','risk_tier','primary_proxy_target','is_known_proxy']].copy()
disp.columns = ['Feature','MI (Outcome)','Max Proxy MI','Surrogate Score','Composite Risk','Risk Tier','Proxy For','Known Proxy']
st.dataframe(disp.head(15), use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# Triple justification panel
st.markdown('<div class="card"><div class="card-title">AI Triple Justification Panel (Gemini)</div>', unsafe_allow_html=True)
st.markdown('<div class="box-info">Click "Analyze" on any feature to generate its statistical, moral, and legal justification. Generate one at a time to avoid rate limits.</div>', unsafe_allow_html=True)

top_feats = risk_df[risk_df['risk_tier'].isin(['HIGH','MEDIUM'])].head(5)
for _, fr in top_feats.iterrows():
    feat = fr['feature']; risk_tier = fr['risk_tier']; proxy_target = fr['primary_proxy_target']
    with st.expander(f"{feat.upper()}   |   Risk: {risk_tier}   |   Proxy for: {proxy_target}", expanded=(risk_tier == 'HIGH')):
        jus = gemini_justifications.get(feat)
        if jus is None:
            if st.button("Analyze with Gemini", key=f"btn_jus_{feat}"):
                with st.spinner(f"Generating justification for {feat}..."):
                    ctx = f"Dataset with {len(df)} records for {domain} decisions. Target: {target_col}. Sensitive: {', '.join(sensitive_cols)}."
                    res = generate_triple_justification(feature_name=feat, mi_score=fr['mi_with_outcome'], proxy_target=proxy_target, domain=domain, dataset_context=ctx)
                    st.session_state.gemini_justifications[feat] = res
                    st.rerun()
        else:
            rec = jus.get('recommended_action','')
            if rec:
                st.markdown(f'<div class="box-warn"><strong>Recommended Action:</strong> {rec}</div>', unsafe_allow_html=True)
            jc1, jc2, jc3 = st.columns(3)
            with jc1:
                st.markdown(f'<div class="jp stat"><div class="jp-label">Statistical Basis</div><div class="jp-text">{jus.get("statistical","No data.")}</div></div>', unsafe_allow_html=True)
            with jc2:
                st.markdown(f'<div class="jp moral"><div class="jp-label">Moral / Historical Context</div><div class="jp-text">{jus.get("moral_historical","No data.")}</div></div>', unsafe_allow_html=True)
            with jc3:
                st.markdown(f'<div class="jp legal"><div class="jp-label">Legal / Regulatory Basis</div><div class="jp-text">{jus.get("legal","No data.")}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Intersectional
intersectional = sensitivity_results.get('intersectional_risks', [])
if intersectional:
    st.markdown('<div class="mod-header" style="margin-top:1rem;"><span class="mod-num">BLIND SPOTS</span><span class="mod-title">Intersectional Risk Detection</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="box-info">These feature pairs reveal sensitive attribute information in combination, even when each appears safe individually.</div>', unsafe_allow_html=True)
    for risk in intersectional[:5]:
        blind = '<span class="b-fail">BLIND SPOT</span>' if risk.get('blind_spot') else ''
        st.markdown(f'<div class="isect-card"><div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.4rem;"><span style="font-family: Roboto Mono, monospace; font-size:0.9rem; font-weight:600; color:#6E1A37;">{risk["feature_1"]} x {risk["feature_2"]}</span><span style="color:#72BAA9; font-size:0.8rem;">proxy for {risk["sensitive_attr"]}</span>{blind}</div><div style="font-family: Roboto Mono, monospace; font-size:0.78rem; color:#6E1A37; display:flex; gap:1.5rem; flex-wrap:wrap;"><span>MI f1: {risk["mi_f1_alone"]}</span><span>MI f2: {risk["mi_f2_alone"]}</span><span>MI combined: {risk["mi_combined"]}</span><span style="font-weight:600; color:#AE2448;">Synergy: +{risk["synergy_score"]}</span></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ===================================================================
# MODULE 02 -- FAIRNESS CONFLICT VISUALIZER
# ===================================================================

st.markdown('<div class="mod-header"><span class="mod-num">MODULE 02</span><span class="mod-title">Fairness Conflict Visualizer</span></div>', unsafe_allow_html=True)

# Metric reference bar
refs_html = "".join([
    f'<a class="metric-ref" href="{info["paper"]}" target="_blank" title="{info["full"]} -- {info["paper_label"]}">{info["short"]}</a>&nbsp;&nbsp;'
    for info in METRIC_LINKS.values()
])
st.markdown(f'<div class="surface" style="padding: 0.9rem 1.4rem;"><span style="font-size:0.82rem; color:#6E1A37; font-weight:500;">Metrics: </span>{refs_html}<span style="font-size:0.74rem; color:#AE2448; margin-left:0.5rem;">-- click abbreviation to view source paper</span></div>', unsafe_allow_html=True)

# Radar + heatmap side by side in white cards
rc1, rc2 = st.columns(2)
with rc1:
    st.markdown('<div class="surface">', unsafe_allow_html=True)
    st.plotly_chart(plot_fairness_radar(agg_metrics, thresholds), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with rc2:
    st.markdown('<div class="surface">', unsafe_allow_html=True)
    st.plotly_chart(plot_conflict_heatmap(agg_metrics, fairness_results.get('detected_conflicts', [])), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Group comparison + trade-off side by side
gc1, gc2 = st.columns(2)
with gc1:
    st.markdown('<div class="surface">', unsafe_allow_html=True)
    group_metrics_data = fairness_results.get('group_metrics', {})
    if group_metrics_data:
        metric_options = ['positive_rate','true_positive_rate','false_positive_rate','positive_predictive_value','accuracy']
        sel = st.selectbox("Group Comparison Metric", metric_options, format_func=lambda x: x.replace('_',' ').title())
        st.plotly_chart(plot_group_comparison(group_metrics_data, sel), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with gc2:
    st.markdown('<div class="surface">', unsafe_allow_html=True)
    if tradeoffs:
        st.plotly_chart(plot_tradeoff_chart(tradeoffs, agg_metrics), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Metrics detail table
st.markdown('<div class="card"><div class="card-title">Fairness Metrics Detail</div>', unsafe_allow_html=True)
for metric, val in agg_metrics.items():
    is_pass   = abs(val) <= thresholds.get(metric, 0.1)
    badge     = '<span class="b-pass">PASS</span>' if is_pass else '<span class="b-fail">FAIL</span>'
    desc      = METRIC_DESCRIPTIONS.get(metric, {})
    minfo     = METRIC_LINKS.get(metric, {})
    plink     = f'<a class="metric-ref" href="{minfo["paper"]}" target="_blank">{minfo["paper_label"]}</a>' if minfo else ''
    st.markdown(f'<div class="metric-row"><div class="metric-name"><strong>{desc.get("name", metric)}</strong> <span style="color:#AE2448; font-size:0.75rem;">({desc.get("short","")})</span><div style="font-size:0.78rem; color:#6E1A37; opacity:0.75; margin-top:0.1rem;">{desc.get("definition","")}</div><div style="margin-top:0.2rem;">{plink}</div></div><div class="metric-val">{val:.4f}</div>{badge}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Conflicts
conflicts = fairness_results.get('detected_conflicts', [])
if conflicts:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="color:#AE2448; border-bottom-color:#AE2448;">Detected Fairness Conflicts</div>', unsafe_allow_html=True)
    for c in conflicts:
        st.markdown(f'<div class="conflict-box"><div class="conflict-title">[{c["conflict_type"]}] {c["metric_1"].replace("_"," ").upper()} vs {c["metric_2"].replace("_"," ").upper()}</div><div class="conflict-desc">{c["description"]}</div><div style="margin-top:0.35rem; font-size:0.75rem; font-family: Roboto Mono, monospace; color:#AE2448;">Severity: {c["severity"]} &nbsp;|&nbsp; M1: {c["metric_1_value"]:.4f} &nbsp;|&nbsp; M2: {c["metric_2_value"]:.4f}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Rationale selection
st.markdown('<div class="card"><div class="card-title">Select Fairness Criterion and Generate Audit Rationale</div>', unsafe_allow_html=True)
metric_display_names = {m: METRIC_DESCRIPTIONS[m]['name'] for m in agg_metrics}
chosen_metric_name = st.selectbox("Choose the fairness definition your organization will prioritize:", list(metric_display_names.values()))
chosen_metric_key  = [k for k,v in metric_display_names.items() if v == chosen_metric_name][0]
user_justification = st.text_area("Optional: Add your organization's reasoning:", placeholder="e.g., Our legal team determined that Equal Opportunity is most aligned with ECOA requirements...", height=80)
if st.button("Generate Audit Rationale via Gemini", type="secondary"):
    with st.spinner("Generating rationale..."):
        rationale = generate_fairness_rationale(
            chosen_metric=chosen_metric_name, metric_values=agg_metrics, domain=domain,
            sensitive_attribute=fairness_results.get('sensitive_col','sensitive attribute'),
            accuracy_cost=tradeoffs.get(chosen_metric_key,{}).get('accuracy_cost_pct', 0.0)
        )
        if user_justification:
            rationale += "\n\nOrganization Note: " + user_justification
        st.session_state.chosen_metric    = chosen_metric_key
        st.session_state.gemini_rationale = rationale
if st.session_state.gemini_rationale:
    ts = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    st.markdown(f'<div class="gemini-out"><div class="gemini-label">AI-Generated Audit Rationale (Gemini) -- {ts}</div>{st.session_state.gemini_rationale}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ===================================================================
# MODULE 03 -- BLACK-BOX PROBE
# ===================================================================

st.markdown('<div class="mod-header"><span class="mod-num">MODULE 03</span><span class="mod-title">Black-Box Counterfactual Probe</span></div>', unsafe_allow_html=True)

if blackbox_results:
    if blackbox_results.get('overall_bias_detected'):
        st.markdown('<div class="box-error"><strong>BIAS DETECTED:</strong> Counterfactual testing found that changing proxy features significantly alters model outcomes for otherwise identical records. This indicates the model may be discriminating based on demographic signals embedded in non-sensitive features.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="box-success"><strong>No Significant Bias Detected:</strong> Counterfactual tests did not reveal systematic outcome changes attributable to proxy feature manipulation.</div>', unsafe_allow_html=True)

    bc1, bc2, bc3 = st.columns(3)
    rc = '#AE2448' if blackbox_results['combined_bias_rate'] > 0.10 else '#72BAA9'
    dc = '#AE2448' if blackbox_results['combined_mean_prob_delta'] > 0.05 else '#72BAA9'
    bc1.markdown(f'<div class="kpi-card"><div class="stat-num">{blackbox_results["total_tests_run"]:,}</div><div class="stat-label">Counterfactual Tests</div></div>', unsafe_allow_html=True)
    bc2.markdown(f'<div class="kpi-card"><div class="stat-num" style="color:{rc};">{blackbox_results["combined_bias_rate"]*100:.1f}%</div><div class="stat-label">Combined Proxy Bias Rate</div></div>', unsafe_allow_html=True)
    bc3.markdown(f'<div class="kpi-card"><div class="stat-num" style="color:{dc};">{blackbox_results["combined_mean_prob_delta"]:.4f}</div><div class="stat-label">Mean Probability Shift</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    feat_summary = blackbox_results.get('feature_impact_summary', {})
    if feat_summary:
        bb1, bb2 = st.columns([3, 2])
        with bb1:
            st.markdown('<div class="surface">', unsafe_allow_html=True)
            st.plotly_chart(plot_blackbox_impact(feat_summary), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with bb2:
            st.markdown('<div class="card"><div class="card-title">Per-Feature Bias Breakdown</div>', unsafe_allow_html=True)
            for feat, summary in feat_summary.items():
                badge = '<span class="b-fail">BIAS</span>' if summary.get('bias_detected') else '<span class="b-pass">CLEAR</span>'
                st.markdown(f'<div class="metric-row"><div class="metric-name" style="font-size:0.88rem;"><strong>{feat}</strong><div style="font-size:0.75rem; color:#6E1A37; margin-top:0.1rem;">Change rate: {summary["outcome_change_rate"]*100:.1f}% &nbsp;|&nbsp; Mean delta: {summary["mean_prob_delta"]:.4f} &nbsp;|&nbsp; Max: {summary["max_prob_delta"]:.4f}</div></div>{badge}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    combined_probes = blackbox_results.get('combined_probe_results', [])
    if combined_probes:
        st.markdown('<div class="card"><div class="card-title">Sample Multi-Feature Counterfactual Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="box-info" style="margin-bottom:0.75rem;">These tests simultaneously changed multiple proxy features to simulate a complete demographic shift, revealing compounded bias effects.</div>', unsafe_allow_html=True)
        changed_count = sum(1 for r in combined_probes if r['outcome_changed'])
        for i, result in enumerate(combined_probes[:5]):
            chstr  = " &nbsp;|&nbsp; ".join([f"{c['feature']}: {c['original']} -> {c['changed']}" for c in result['feature_changes']])
            obadge = '<span class="b-fail">CHANGED</span>' if result['outcome_changed'] else '<span class="b-pass">STABLE</span>'
            dcolor = '#AE2448' if abs(result['probability_delta']) > 0.05 else '#72BAA9'
            delta  = f"{result['probability_delta']:+.4f}"
            st.markdown(f'<div class="cf-row"><span class="cf-idx">Test {i+1}</span><span style="flex:1; color:#6E1A37;">{chstr}</span><span class="cf-delta" style="color:{dcolor};">{delta}</span>{obadge}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="padding:0.6rem 0.8rem; font-size:0.85rem; color:#6E1A37;"><strong>{changed_count} of {len(combined_probes)}</strong> combined probe tests resulted in outcome changes.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if combined_probes and any(r['outcome_changed'] for r in combined_probes[:3]):
        if st.button("Generate Gemini Interpretation of Counterfactual Findings", type="secondary"):
            with st.spinner("Interpreting results..."):
                sr = next((r for r in combined_probes if r['outcome_changed']), combined_probes[0])
                interp = generate_blackbox_interpretation(
                    feature_changes=sr['feature_changes'],
                    original_outcome='positive outcome' if sr['original_prediction'] == 1 else 'negative outcome',
                    new_outcome='positive outcome' if sr['counterfactual_prediction'] == 1 else 'negative outcome',
                    domain=domain
                )
                st.markdown(f'<div class="gemini-out"><div class="gemini-label">AI Counterfactual Interpretation (Gemini)</div>{interp}</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ===================================================================
# EXPORT
# ===================================================================

st.markdown('<div class="mod-header"><span class="mod-num">EXPORT</span><span class="mod-title">Audit Report Export</span></div>', unsafe_allow_html=True)
st.markdown('<div class="card"><div class="card-title">Generate PDF Audit Report</div>', unsafe_allow_html=True)
st.markdown('<div class="box-info">The audit report includes all findings, metric values, conflict analysis, justification text, and timestamps suitable for regulatory documentation.</div>', unsafe_allow_html=True)

ex1, ex2 = st.columns(2)
with ex1:
    if st.button("Generate and Download PDF Audit Report", type="primary", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            try:
                from modules.report_generator import generate_audit_report
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    report_path = tmp.name
                generate_audit_report(
                    domain=domain, sensitive_cols=sensitive_cols, target_col=target_col,
                    sensitivity_results=sensitivity_results, fairness_results=fairness_results,
                    blackbox_results=blackbox_results,
                    chosen_metric=st.session_state.chosen_metric or 'Not selected',
                    gemini_rationale=st.session_state.gemini_rationale or '',
                    gemini_summary=st.session_state.gemini_summary or '',
                    output_path=report_path
                )
                with open(report_path, 'rb') as f:
                    pdf_bytes = f.read()
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button("Download PDF Report", data=pdf_bytes, file_name=f"fairscope_audit_{ts}.pdf", mime='application/pdf', use_container_width=True)
                st.markdown('<div class="box-success">PDF ready. Click above to download.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="box-error">PDF generation failed: {str(e)}</div>', unsafe_allow_html=True)

with ex2:
    if st.button("Export Raw Findings as JSON", type="secondary", use_container_width=True):
        export_data = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'domain': domain, 'target_col': target_col, 'sensitive_cols': sensitive_cols,
            'sensitivity_summary': {
                'high_risk_count': sensitivity_results['high_risk_count'],
                'medium_risk_count': sensitivity_results['medium_risk_count'],
                'low_risk_count': sensitivity_results['low_risk_count'],
                'top_features': sensitivity_results['risk_dataframe'].head(10).to_dict('records')
            },
            'fairness_metrics': {
                'aggregate': fairness_results['aggregate_metrics'],
                'overall_accuracy': fairness_results['overall_accuracy'],
                'conflicts_detected': n_conflicts,
            },
            'blackbox_results': {
                'overall_bias_detected': blackbox_results.get('overall_bias_detected'),
                'combined_bias_rate': blackbox_results.get('combined_bias_rate'),
                'total_tests': blackbox_results.get('total_tests_run')
            },
            'chosen_metric': st.session_state.chosen_metric,
            'gemini_rationale': st.session_state.gemini_rationale
        }
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button("Download JSON", data=json.dumps(export_data, indent=2, default=str), file_name=f"fairscope_findings_{ts}.json", mime='application/json', use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="disclaimer"><strong>Disclaimer:</strong> Fairscope is an automated analysis tool and does not constitute legal advice. AI-generated justifications (Gemini) may not reflect current law and should be reviewed by qualified legal counsel. Fairness metrics are statistical measures and do not guarantee regulatory compliance. Organizations remain solely responsible for their AI systems compliance with applicable law. No personally identifiable data is transmitted or stored by Fairscope beyond your local session.</div>', unsafe_allow_html=True)