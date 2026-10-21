# Retain — Customer Churn & Feedback Intelligence

A clean, Vercel-ready customer analytics application that combines churn classification, probability calibration, complaint mining, cohort analysis, segmentation, and ranked retention recommendations.

## What it does

- **Calibrated churn prediction:** Logistic Regression with sigmoid calibration (`CalibratedClassifierCV`). Scores are probabilities, not just labels.
- **Unseen-customer evaluation:** A stratified 25% holdout of labelled customers is set aside for reporting ROC AUC, average precision, accuracy, precision, recall, F1, Brier score, and expected calibration error.
- **Baseline comparison:** Random Forest metrics are shown alongside the calibrated logistic model. The baseline's probabilities are explicitly labelled **not calibrated**.
- **Feedback mining:** VADER-style sentiment estimates, TF-IDF text features, and topic extraction using TF-IDF + NMF when enough varied text is available.
- **Retention action ranking:** Rule-based suggestions use churn risk plus feedback and account signals (billing/price, support, reliability, payment friction, contract type, usage, tenure).
- **Cohort analysis:** Signup-month cohorts when a date is available; otherwise tenure bands when a tenure feature is detected.
- **Customer segmentation:** KMeans clusters based on available numeric account features.
- **Scoring new customers:** Rows with a blank churn target are allowed. They are excluded from training/evaluation and then scored by the model refit on all labelled history.
- **Downloadable results:** Export each customer's risk score, risk tier, feedback signal, segment, and recommended action as CSV.

## Colab companion notebook

`notebooks/churn_feedback_mining_colab.ipynb` provides a reproducible exploration workflow: dataset checks, target mapping, TF-IDF + structured preprocessing, calibrated Logistic Regression, Random Forest baseline, held-out metrics, calibration curve, VADER sentiment, NMF topics, cohort summaries, KMeans segments, action hypotheses and CSV export. Upload the sample CSV or your own labelled customer file to Colab.

## Project structure

```text
churn-feedback-studio/
├── app.py                       # FastAPI API (Vercel recognizes this entry point)
├── requirements.txt
├── requirements-dev.txt
├── vercel.json
├── .python-version
├── notebooks/
│   └── churn_feedback_mining_colab.ipynb
├── src/
│   └── churn_engine.py          # ML + NLP + analytics pipeline
├── public/
│   ├── index.html               # UI
│   ├── styles.css               # Clean, cream-and-sage theme
│   ├── app.js                   # Upload, charts, tables, CSV export
│   └── sample-churn-data.csv    # Synthetic sample data
├── sample_data/
│   └── demo_customers.csv       # Same synthetic dataset for local experiments
└── tests/
```

## Quick start locally

Use Python 3.12 for the closest match to the Vercel runtime.

### Windows / PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. OpenAPI documentation is at `http://127.0.0.1:8000/docs` and the health endpoint is `http://127.0.0.1:8000/api/health`.

## Deploy to Vercel

1. Download/extract the project or push this folder to a GitHub repository.
2. Keep `app.py`, `requirements.txt`, `vercel.json`, `src/`, and `public/` at the repository root (do not upload only the nested folder).
3. Import the repository at [Vercel New Project](https://vercel.com/new).
4. Set the Root Directory to the folder containing `app.py` and `public/`.
5. Leave custom Build Command and Output Directory blank. Vercel's Python runtime discovers the FastAPI `app` in `app.py`; assets in `public/` are served by Vercel's CDN.
6. Deploy, then test:
   - `/` — application UI
   - `/api/health` — JSON health status
   - `/api/info` — feature list and limits
   - `/docs` — API documentation

The application intentionally does **not** mount `StaticFiles` on Vercel, to avoid shadowing CDN-served `public/` files. The mount is enabled only for local development.

## Upload schema

A CSV must include a **binary churn target** with two observed classes. Common examples: `churn` containing `0/1`, or `Churn` containing `Yes/No`. The interface previews the column names and lets you select the target and feedback text field.

Helpful optional columns:

| Example column | Used for |
|---|---|
| `customer_id` | Human-readable ID in the results table; excluded from model features when selected/detected |
| `tenure_months` | Segments, retention actions, and fallback tenure cohorts |
| `monthly_charges` | Model feature and plan-value action heuristics |
| `support_tickets_90d` | Model feature and service follow-up heuristics |
| `payment_delay_count` | Model feature and billing assistance heuristics |
| `contract_type` | Model feature and contract action heuristics |
| `signup_date` | Cohort analysis by signup month; excluded as a raw model feature |
| `complaint_text` | TF-IDF churn model feature, sentiment, topic extraction and action clues |
| `churn` | Historical target. Leave blank for customers that should be scored but have no observed outcome yet. |

The uploaded data should be representative of the customers you ultimately want to score. For meaningful evaluation, use historical examples with known outcomes and avoid using columns that reveal a churn decision made after the prediction date.

## Vercel-friendly limits

- Maximum upload file size in this build: **3 MiB**. Vercel Functions have a 4.5 MB request/response payload limit; the app deliberately uses a lower cap to leave room for multipart/form overhead.
- Maximum records: **2,500** per analysis.
- Maximum columns: **80**. The Vercel-friendly model pipeline uses up to 18 numeric columns, 18 categorical columns, and one selected text column to bound training cost; the API response lists any ignored predictor columns.
- Use at least 50 labelled historical customers with at least 12 examples in each churn class for the holdout + 3-fold calibration workflow. You can include unlabelled rows alongside them.
- Larger workloads should run in Colab or a persistent backend designed for batch processing. Vercel functions are not a long-running model-training worker.

## Methods and honest interpretation

- **Primary model:** Calibrated Logistic Regression with text TF-IDF plus imputed numeric features and one-hot encoded categorical features.
- **Evaluation:** 75/25 stratified train/test split among labelled rows. Metrics shown in the UI come from the held-out test partition and represent an estimate of performance on unseen customers under a random-split assumption.
- **Scoring:** After holdout evaluation, a calibrated model is refit using all available labelled customers to score every row, including new rows with blank churn labels. These scores are used for ranking work, not to re-report the holdout metric.
- **Probability quality:** Brier score and expected calibration error are included. Lower is better. Calibration on a random split does not guarantee calibration under customer or time drift.
- **Text sentiment:** VADER-style lexicon scoring works best with English text. It is not a fine-tuned complaint classifier, and sarcasm/context may confuse it.
- **Topic extraction:** NMF topics are discovered from the uploaded corpus. Topic names are the most representative terms and need human interpretation.
- **Retention actions:** The ranked recommendations are heuristic rules. Their priority scores are **not** estimates of causal uplift, money saved, or churn reduction. Validate action effectiveness with controlled experiments or uplift modelling before business use.
- **Segmentation:** KMeans segments depend on the numeric features and scaling in the uploaded data; segment IDs have no inherent order or stable meaning across separate runs.

## Privacy and security

The app does not save uploaded CSVs in an application database or persist trained models between requests. The file contents are sent to the Vercel-hosted API for processing. Use synthetic or anonymised data for demos. Before using customer PII or regulated data, review your organisation's data handling rules, vendor configuration, access controls, and retention requirements.

## Run tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## License and intended use

Provided as an educational/portfolio prototype. Confirm your dataset's permissions and applicable privacy requirements. Churn estimates and retention recommendations are decision-support signals; keep a human in the loop.
