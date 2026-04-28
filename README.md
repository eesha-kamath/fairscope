# Fairscope - AI-Powered Fairness Auditor

Fairscope is a hackathon MVP for auditing automated decision systems
in hiring, lending, and healthcare for hidden bias and discriminatory proxies.

---

## Modules

- **Module 1 - Attribute Sensitivity Scorer**: Detects proxy features using mutual information,
  surrogate model scoring, and intersectional blind-spot detection. Generates triple
  justifications (statistical, moral/historical, legal) via Google Gemini.

- **Module 2 - Fairness Conflict Visualizer**: Computes 5 fairness metrics (Demographic Parity,
  Equalized Odds, Equal Opportunity, Predictive Parity, Calibration), detects mathematical
  conflicts, and generates auditable rationale for the chosen criterion.

- **Module 3 - Black-Box Probe**: Systematic counterfactual testing to audit models without
  internal access. Tests if proxy feature changes shift model outcomes.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your Gemini API key

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at: https://aistudio.google.com/app/apikey

### 3. Generate the sample dataset

```bash
python generate_sample_data.py
```

This creates `data/adult_income_sample.csv` -- a synthetic Adult Income dataset
with known gender and race proxies baked in (marital_status, relationship, occupation).

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Usage

1. Upload a CSV or check "Use built-in sample dataset" in the sidebar
2. Enter your Gemini API key in the sidebar (or set it in .env)
3. Set: Domain (hiring/lending/healthcare/insurance), Target variable, Sensitive attributes
4. Click "RUN FULL AUDIT"
5. Review findings across all three modules
6. Select a fairness criterion and generate an audit rationale
7. Export the PDF audit report

---

## Tech Stack

- **Frontend/UI**: Streamlit
- **Data**: Pandas, NumPy
- **ML/Metrics**: scikit-learn (MI, surrogate models, GBM)
- **AI Justifications**: Google Gemini 1.5 Flash via REST API
- **Visualizations**: Plotly
- **PDF Export**: fpdf2

---

## Sample Dataset Notes

The Adult Income sample has the following intentional biases for demo purposes:
- `marital_status` and `relationship` are proxies for `sex`
- `occupation` correlates with both `sex` and `race`
- Income probability is higher for Male records (simulating historical bias)

This allows Fairscope to detect real proxy patterns and conflicts.

---

## Legal Note

This tool does not constitute legal advice. AI justifications should be reviewed
by qualified legal counsel. See in-app disclaimer for full terms.
