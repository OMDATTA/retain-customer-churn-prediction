"""Customer churn prediction and complaint-mining engine.

This module is deliberately designed for a small, stateless Vercel function:
- bounded CSV size and row count
- calibrated logistic-regression churn probabilities
- a compact random-forest comparison baseline
- VADER-style sentiment, NMF topics, cohort and KMeans summaries
- heuristic retention-action ranking (not causal uplift estimates)
"""
from __future__ import annotations

import re
import warnings
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import NMF

MAX_ROWS = 2500
MAX_COLUMNS = 80
MAX_TEXT_FEATURES = 500

TARGET_CANDIDATES = (
    "churn", "churned", "is_churn", "churn_flag", "exited", "attrition",
    "left", "cancelled", "canceled", "will_churn", "churn_status", "target",
)
TEXT_CANDIDATES = (
    "complaint", "complaint_text", "feedback", "feedback_text", "customer_feedback",
    "customer_complaint", "review", "review_text", "comment", "comments", "message",
    "support_notes", "ticket_text", "reason", "text",
)
ID_CANDIDATES = ("customer_id", "customerid", "client_id", "account_id", "user_id", "id")
DATE_CANDIDATES = ("signup_date", "join_date", "start_date", "registration_date", "created_at", "customer_since", "cohort_date")

POSITIVE_LABELS = {"yes", "y", "true", "churn", "churned", "left", "exited", "cancelled", "canceled", "lost", "attrited", "inactive", "1", "will churn", "will_churn"}
NEGATIVE_LABELS = {"no", "n", "false", "stay", "stayed", "retained", "active", "0", "not churn", "not_churn", "existing", "loyal"}

_POSITIVE_WORDS = {
    "good", "great", "excellent", "helpful", "happy", "love", "resolved", "quick", "thanks", "thank", "easy", "reliable", "satisfied", "perfect", "friendly", "improved", "appreciate", "smooth", "recommend", "pleasant"
}
_NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "hate", "angry", "slow", "broken", "late", "delay", "delayed", "refund", "cancel", "expensive", "charged", "charge", "bill", "billing", "issue", "problem", "unhelpful", "unhappy", "frustrated", "disappointed", "complaint", "wrong", "error", "worse", "unresolved", "wait", "waiting"
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def guess_column(columns: list[str], candidates: tuple[str, ...], contains: tuple[str, ...] = ()) -> str | None:
    normalised = {_norm(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    for key, original in normalised.items():
        if any(piece in key for piece in contains):
            return original
    return None


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return value if isinstance(value, (str, int, bool)) else str(value)


def _empty_as_missing(series: pd.Series) -> pd.Series:
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        cleaned = series.astype("string").str.strip()
        return cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA, "NA": pd.NA, "N/A": pd.NA})
    return series


def encode_churn_target(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """Map a binary target to 0/1, where 1 denotes churn/positive class.

    Numeric two-class targets use the larger class as positive unless the classes
    are exactly 0 and 1. Text targets use common churn labels first; otherwise
    the lexically last label is treated as positive and the choice is reported.
    """
    raw = _empty_as_missing(series)
    present = raw.notna()
    yraw = raw[present]
    if yraw.nunique(dropna=True) != 2:
        raise ValueError(f"The selected target must contain exactly two labels among labelled rows; found {yraw.nunique(dropna=True)}.")

    numeric = pd.to_numeric(yraw, errors="coerce")
    if numeric.notna().all() and numeric.nunique() == 2:
        vals = sorted(numeric.unique().tolist())
        positive_val = 1 if vals == [0, 1] else vals[-1]
        encoded = (numeric == positive_val).astype(int)
        result = pd.Series(index=series.index, dtype="float64")
        result.loc[yraw.index] = encoded.to_numpy()
        return result, {"positive_label": str(positive_val), "negative_label": str(vals[0] if vals[1] == positive_val else vals[1]), "mapping_note": ("Numeric 0/1 labels; 1 is treated as churn." if vals == [0, 1] else "Non-0/1 numeric labels; the larger value was assumed to mean churn. Verify this assumption before operational use.")}

    labels = yraw.astype(str).str.strip()
    unique_labels = sorted(labels.unique().tolist(), key=lambda x: x.lower())
    norm_map = {_norm(v).replace("_", " "): v for v in unique_labels}
    positive_match = None
    negative_match = None
    for normalized, original in norm_map.items():
        simple = normalized.strip()
        if simple in POSITIVE_LABELS or ("churn" in simple and not simple.startswith("not ")) or simple in {"exit", "exited"}:
            positive_match = original
        if simple in NEGATIVE_LABELS or simple.startswith("not churn"):
            negative_match = original
    if positive_match is not None and positive_match in unique_labels:
        if negative_match is None or negative_match == positive_match:
            negative_match = next(v for v in unique_labels if v != positive_match)
    elif negative_match is not None and negative_match in unique_labels:
        positive_match = next(v for v in unique_labels if v != negative_match)
    else:
        negative_match, positive_match = unique_labels[0], unique_labels[1]
        note = "Unrecognised text labels; the second sorted label was treated as churn. Verify the positive class before using operationally."
        encoded = (labels == positive_match).astype(int)
        result = pd.Series(index=series.index, dtype="float64")
        result.loc[yraw.index] = encoded.to_numpy()
        return result, {"positive_label": str(positive_match), "negative_label": str(negative_match), "mapping_note": note}

    encoded = (labels == positive_match).astype(int)
    result = pd.Series(index=series.index, dtype="float64")
    result.loc[yraw.index] = encoded.to_numpy()
    return result, {"positive_label": str(positive_match), "negative_label": str(negative_match), "mapping_note": "Text labels mapped using common churn/retention names."}


def _safe_text(series: pd.Series) -> pd.Series:
    result = series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return result.mask(result.eq(""), "no feedback provided")


def _sentiment_score(text: str) -> tuple[float, str]:
    text = (text or "").strip()
    if not text or text.lower() == "no feedback provided":
        return 0.0, "No feedback"
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        global _VADER
        try:
            analyzer = _VADER
        except NameError:
            _VADER = SentimentIntensityAnalyzer()
            analyzer = _VADER
        compound = float(analyzer.polarity_scores(text)["compound"])
    except Exception:
        tokens = re.findall(r"[a-z']+", text.lower())
        pos = sum(1 for token in tokens if token in _POSITIVE_WORDS)
        neg = sum(1 for token in tokens if token in _NEGATIVE_WORDS)
        compound = float((pos - neg) / max(1, pos + neg))
    label = "Negative" if compound <= -0.05 else ("Positive" if compound >= 0.05 else "Neutral")
    return compound, label


def _sentiment_rows(texts: pd.Series) -> tuple[list[float], list[str]]:
    scored = [_sentiment_score(t) for t in texts.tolist()]
    return [x[0] for x in scored], [x[1] for x in scored]


def _text_topics(texts: pd.Series) -> tuple[list[str | None], list[dict[str, Any]]]:
    """Lightweight topic extraction; gracefully reports when text is too sparse."""
    cleaned = texts.fillna("").astype(str).tolist()
    valid_indices = [i for i, text in enumerate(cleaned) if text.strip() and text.lower() != "no feedback provided"]
    nonempty = [cleaned[i] for i in valid_indices]
    empty_assignments: list[str | None] = [None] * len(cleaned)
    if len(nonempty) < 12 or len(set(nonempty)) < 5:
        return empty_assignments, []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=350, ngram_range=(1, 2), min_df=2, max_df=0.95)
        matrix = vectorizer.fit_transform(nonempty)
        if matrix.shape[1] < 4:
            return empty_assignments, []
        n_topics = min(5, max(2, len(nonempty) // 35), matrix.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nmf = NMF(n_components=n_topics, init="nndsvda", max_iter=120, random_state=42)
            document_topics = nmf.fit_transform(matrix)
        terms = vectorizer.get_feature_names_out()
        topics = []
        topic_names: list[str] = []
        for i, row in enumerate(nmf.components_):
            top_terms = [str(terms[j]) for j in row.argsort()[-5:][::-1]]
            name = " · ".join(top_terms[:3])
            topic_names.append(name)
            topics.append({"topic": f"Topic {i + 1}", "label": name, "keywords": top_terms})
        assignments_idx = document_topics.argmax(axis=1)
        for index, topic_index in zip(valid_indices, assignments_idx):
            empty_assignments[index] = f"Topic {int(topic_index) + 1}: {topic_names[int(topic_index)]}"
        return empty_assignments, topics
    except Exception:
        return empty_assignments, []


def _make_preprocessor(X: pd.DataFrame, text_column: str | None) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    available = list(X.columns)
    texts = [text_column] if text_column and text_column in available else []
    numeric = [c for c in available if c not in texts and pd.api.types.is_numeric_dtype(X[c])][:18]
    categorical = [c for c in available if c not in texts and not pd.api.types.is_numeric_dtype(X[c])][:18]
    transformers = []
    if texts:
        transformers.append(("text", TfidfVectorizer(max_features=MAX_TEXT_FEATURES, ngram_range=(1, 2), min_df=1, strip_accents="unicode", sublinear_tf=True), texts[0]))
    if numeric:
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=25)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    if not transformers:
        raise ValueError("No usable predictor columns remain. Choose a feedback/text field or include numeric/categorical customer features.")
    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.25)
    return preprocessor, numeric, categorical, texts


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    labels = (probability >= 0.5).astype(int)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "expected_calibration_error": float(_ece(y_true, probability)),
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
        result["average_precision"] = float(average_precision_score(y_true, probability))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def _ece(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = max(1, len(y_true))
    score = 0.0
    for i in range(bins):
        mask = (prob >= edges[i]) & (prob <= edges[i + 1] if i == bins - 1 else prob < edges[i + 1])
        if np.any(mask):
            score += float(mask.sum()) / total * abs(float(y_true[mask].mean()) - float(prob[mask].mean()))
    return float(score)


def _calibration_bins(y_true: np.ndarray, prob: np.ndarray, bins: int = 8) -> list[dict[str, Any]]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        mask = (prob >= edges[i]) & (prob <= edges[i + 1] if i == bins - 1 else prob < edges[i + 1])
        if np.any(mask):
            rows.append({
                "bucket": f"{edges[i]:.2f}–{edges[i + 1]:.2f}",
                "mean_predicted": float(prob[mask].mean()),
                "observed_churn": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            })
    return rows


def _feature_importance(model: Any) -> list[dict[str, Any]]:
    try:
        calibrated_classifiers = getattr(model, "calibrated_classifiers_", [])
        all_coefs, names = [], None
        for calibrated_item in calibrated_classifiers:
            estimator = getattr(calibrated_item, "estimator", None)
            if estimator is None:
                continue
            pre = estimator.named_steps["features"]
            clf = estimator.named_steps["classifier"]
            names = pre.get_feature_names_out()
            coef = np.abs(np.asarray(clf.coef_).reshape(-1))
            if len(coef) == len(names):
                all_coefs.append(coef)
        if not all_coefs or names is None:
            return []
        importance = np.mean(np.vstack(all_coefs), axis=0)
        order = np.argsort(importance)[-14:][::-1]
        results = []
        for idx in order:
            raw = str(names[idx])
            feature = re.sub(r"^(text|numeric|categorical)__", "", raw).replace("_", " ").strip()
            results.append({"feature": feature[:90], "importance": float(importance[idx])})
        max_value = max((r["importance"] for r in results), default=1.0) or 1.0
        for result in results:
            result["relative_percent"] = round(100.0 * result["importance"] / max_value, 1)
        return results
    except Exception:
        return []


def _risk_tier(probability: float) -> str:
    if probability >= 0.75:
        return "Critical"
    if probability >= 0.55:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def _find_col(columns: list[str], tokens: tuple[str, ...]) -> str | None:
    for col in columns:
        name = _norm(col)
        if any(tok in name for tok in tokens):
            return col
    return None


def _row_retention_actions(row: pd.Series, probability: float, sentiment: float, text: str, numeric_cols: list[str], medians: dict[str, float]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    risk_factor = 0.62 + 0.38 * float(probability)
    norm_text = (text or "").lower()

    def add(name: str, reason: str, base: float) -> None:
        if any(a["action"] == name for a in actions):
            return
        actions.append({"action": name, "reason": reason, "priority_score": round(100 * base * risk_factor, 1)})

    if probability >= 0.55:
        add("Proactive retention outreach", "Elevated calibrated churn probability", 0.92 if probability >= 0.75 else 0.78)
    if sentiment < -0.05:
        add("Service recovery call", "Written feedback has a negative sentiment score", 0.94)
    if any(token in norm_text for token in ("bill", "billing", "price", "expensive", "cost", "charged", "fee", "payment")):
        add("Billing and plan review", "Feedback mentions price, billing, charges or payment", 0.87)
    if any(token in norm_text for token in ("support", "agent", "wait", "waiting", "ticket", "response", "unresolved")):
        add("Dedicated support follow-up", "Feedback indicates a support or response issue", 0.89)
    if any(token in norm_text for token in ("outage", "slow", "broken", "error", "crash", "quality", "not working")):
        add("Product reliability follow-up", "Feedback points to a reliability or performance issue", 0.86)

    contract_col = _find_col(list(row.index), ("contract", "plan_type", "subscription_term"))
    if contract_col:
        value = str(row.get(contract_col, "")).lower().replace("_", " ")
        if "month" in value or "monthly" in value or "rolling" in value:
            add("Offer a suitable commitment plan", "Current contract appears short-term or month-to-month", 0.70)

    for col in numeric_cols:
        key = _norm(col)
        try:
            value = float(row.get(col))
        except (TypeError, ValueError):
            continue
        median = medians.get(col)
        if median is None or not np.isfinite(median):
            continue
        if any(t in key for t in ("support_ticket", "ticket_count", "complaint_count", "support_calls", "contact_count")) and value > median and value > 0:
            add("Resolve repeat support contacts", f"{col} is above the dataset median", 0.88)
        if any(t in key for t in ("late_payment", "payment_delay", "overdue", "missed_payment", "failed_payment")) and value > 0:
            add("Offer billing assistance", f"{col} indicates payment friction", 0.82)
        if any(t in key for t in ("monthly_charge", "monthly_charges", "price", "bill_amount", "subscription_fee")) and value > median and any(t in norm_text for t in ("bill", "price", "expensive", "cost", "charge", "fee")):
            add("Review plan-value fit", f"{col} is above the median and feedback mentions cost", 0.84)
        if any(t in key for t in ("usage", "sessions", "logins", "activity")) and value < median:
            add("Product adoption check-in", f"{col} is below the dataset median", 0.67)
        if any(t in key for t in ("tenure", "months_active", "customer_age_months")) and value < median:
            add("New-customer onboarding check-in", f"{col} is below the dataset median", 0.64)

    if not actions:
        add("Review account health", "No single issue dominated; review usage, feedback and service history", 0.52)
    actions.sort(key=lambda a: a["priority_score"], reverse=True)
    return actions[:3]


def _cohort_analysis(df: pd.DataFrame, encoded_y: pd.Series, date_column: str | None, tenure_col: str | None) -> dict[str, Any]:
    work = pd.DataFrame(index=df.index)
    work["y"] = encoded_y
    source = None
    if date_column and date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce", utc=True)
        if dates.notna().sum() >= 8:
            work["cohort"] = dates.dt.strftime("%Y-%m")
            source = f"Signup cohort by {date_column} (UTC month)"
    if source is None and tenure_col and tenure_col in df.columns:
        tenure = pd.to_numeric(df[tenure_col], errors="coerce")
        if tenure.notna().sum() >= 8:
            bins = [-np.inf, 3, 6, 12, 24, 36, 60, np.inf]
            labels = ["≤3 mo", "4–6 mo", "7–12 mo", "13–24 mo", "25–36 mo", "37–60 mo", ">60 mo"]
            work["cohort"] = pd.cut(tenure, bins=bins, labels=labels, include_lowest=True).astype("string")
            source = f"Tenure cohort by {tenure_col}"
    if source is None:
        return {"source": None, "rows": [], "note": "Choose a signup/join date or tenure column to enable cohort analysis."}
    work = work[work["cohort"].notna() & work["y"].notna()]
    grouped = work.groupby("cohort", observed=True)["y"].agg(["count", "mean"]).reset_index()
    if date_column and source.startswith("Signup"):
        grouped = grouped.sort_values("cohort").tail(18)
    rows = [{"cohort": str(r["cohort"]), "customers": int(r["count"]), "churn_rate": float(r["mean"]), "retention_rate": float(1 - r["mean"])} for _, r in grouped.iterrows()]
    return {"source": source, "rows": rows, "note": "Rates use records with observed churn labels only."}


def _segments(df: pd.DataFrame, X_features: pd.DataFrame, encoded_y: pd.Series, probabilities: np.ndarray, numeric_cols: list[str]) -> tuple[list[str | None], list[dict[str, Any]]]:
    candidate_cols = [c for c in numeric_cols if c in X_features.columns and X_features[c].nunique(dropna=True) > 1][:6]
    missing_assignments: list[str | None] = [None] * len(df)
    if len(candidate_cols) < 2 or len(df) < 18:
        return missing_assignments, []
    try:
        matrix = X_features[candidate_cols].apply(pd.to_numeric, errors="coerce")
        matrix = SimpleImputer(strategy="median").fit_transform(matrix)
        scaled = StandardScaler().fit_transform(matrix)
        n_clusters = min(4, max(2, len(df) // 50))
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(scaled)
        segment_names = [f"Segment {int(l) + 1}" for l in labels]
        summaries = []
        for segment in sorted(set(segment_names)):
            idx = np.array([v == segment for v in segment_names])
            positions = np.where(idx)[0]
            y_vals = encoded_y.iloc[positions].dropna()
            summaries.append({
                "segment": segment,
                "customers": int(idx.sum()),
                "average_risk": float(np.mean(probabilities[idx])) if idx.sum() else 0.0,
                "observed_churn_rate": float(y_vals.mean()) if len(y_vals) else None,
                "profile_features": candidate_cols[:4],
            })
        summaries.sort(key=lambda r: r["average_risk"], reverse=True)
        return segment_names, summaries
    except Exception:
        return missing_assignments, []


def _build_demo(n: int = 260, seed: int = 27) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    charges = np.round(rng.normal(68, 23, n).clip(18, 155), 2)
    tickets = rng.poisson(1.2, n)
    payment_delays = rng.poisson(0.45, n)
    usage = np.round(rng.normal(10, 5, n).clip(0.2, 32), 1)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.56, 0.28, 0.16], size=n)
    payment = rng.choice(["Card", "Bank transfer", "Digital wallet", "Direct debit"], size=n, p=[0.34, 0.24, 0.22, 0.20])
    signup_days_ago = rng.integers(20, 1700, n)
    today = pd.Timestamp("2026-10-21", tz="UTC")
    signup_dates = [(today - pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in signup_days_ago]
    latent = (-2.0 + 0.035 * (charges - 65) + 0.42 * tickets + 0.62 * payment_delays - 0.024 * tenure - 0.075 * (usage - 10) + (contract == "Month-to-month") * 0.95)
    prob = 1 / (1 + np.exp(-latent))
    churn = rng.binomial(1, prob)
    negative_text = [
        "I have contacted support twice and the issue is still unresolved.",
        "The monthly bill is too expensive for the service I receive.",
        "Payment failed again and the billing process is frustrating.",
        "The connection is slow and support kept me waiting.",
        "I was charged an unexpected fee and have not received a clear answer.",
        "The service keeps dropping and the last ticket was closed without a fix.",
        "I cannot justify the price after another outage.",
        "The renewal terms were confusing and I want to cancel.",
    ]
    positive_text = [
        "The support agent was helpful and resolved my issue quickly.",
        "The service is reliable and the price feels fair.",
        "Setup was easy and the team followed up with a useful explanation.",
        "I appreciate the quick response and clear billing details.",
        "The new plan works well and I am happy with the experience.",
        "Everything has been smooth and the product is easy to use.",
    ]
    neutral_text = ["I have a question about my current plan.", "Please explain the details of my latest statement.", "I am reviewing the service options for next month."]
    texts = []
    for i in range(n):
        if churn[i] and rng.random() < 0.78:
            texts.append(rng.choice(negative_text))
        elif not churn[i] and rng.random() < 0.68:
            texts.append(rng.choice(positive_text))
        else:
            texts.append(rng.choice(neutral_text + negative_text + positive_text))
    return pd.DataFrame({
        "customer_id": [f"CUST-{10000+i}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_charges": charges,
        "support_tickets_90d": tickets,
        "payment_delay_count": payment_delays,
        "usage_hours_week": usage,
        "contract_type": contract,
        "payment_method": payment,
        "signup_date": signup_dates,
        "complaint_text": texts,
        "churn": churn,
    })


def preview_frame(df: pd.DataFrame) -> dict[str, Any]:
    columns = [str(c) for c in df.columns]
    if not columns:
        raise ValueError("The CSV does not contain a header row or columns.")
    target = guess_column(columns, TARGET_CANDIDATES, ("churn", "attrition", "exited"))
    text = guess_column(columns, TEXT_CANDIDATES, ("complaint", "feedback", "review", "comment", "message", "text"))
    customer_id = guess_column(columns, ID_CANDIDATES, ("customer_id", "account_id", "user_id"))
    date = guess_column(columns, DATE_CANDIDATES, ("signup_date", "join_date", "created_at", "start_date"))
    samples = df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")
    return json_safe({
        "columns": columns,
        "row_count_preview": int(len(df)),
        "target_suggestion": target,
        "text_suggestion": text,
        "customer_id_suggestion": customer_id,
        "date_suggestion": date,
        "sample_rows": samples,
        "max_rows": MAX_ROWS,
        "max_columns": MAX_COLUMNS,
    })


def analyze_customers(
    frame: pd.DataFrame,
    target_column: str,
    text_column: str | None = None,
    customer_id_column: str | None = None,
    date_column: str | None = None,
) -> dict[str, Any]:
    """Fit a calibrated churn model, evaluate a stratified holdout, mine feedback,
    and score every row (including rows with a blank churn label).
    """
    if frame.empty:
        raise ValueError("The CSV has no data rows.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"This Vercel-friendly build supports up to {MAX_ROWS:,} rows per analysis. Use a representative sample or run larger experiments in a notebook.")
    if len(frame.columns) > MAX_COLUMNS:
        raise ValueError(f"The file has {len(frame.columns)} columns; please keep it to {MAX_COLUMNS} or fewer relevant columns.")
    df = frame.copy().reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    if not target_column or target_column not in df.columns:
        raise ValueError("Choose a valid churn target column.")
    target_series = _empty_as_missing(df[target_column])
    encoded_y, label_info = encode_churn_target(target_series)
    labeled_mask = encoded_y.notna()
    labeled_positions = np.where(labeled_mask.to_numpy())[0]
    y_all_labeled = encoded_y.loc[labeled_mask].astype(int)
    class_counts = y_all_labeled.value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < 12:
        raise ValueError("For a reliable stratified holdout and 3-fold calibration, provide at least 12 labelled examples in each churn class (0/retained and 1/churn). Blank target values are allowed for customers you want to score.")
    if len(y_all_labeled) < 50:
        raise ValueError("Provide at least 50 labelled historical customers; blank target rows may be included for scoring new customers.")

    if text_column not in df.columns if text_column else False:
        text_column = None
    if customer_id_column not in df.columns if customer_id_column else False:
        customer_id_column = None
    if date_column not in df.columns if date_column else False:
        date_column = None
    if not customer_id_column:
        customer_id_column = guess_column(list(df.columns), ID_CANDIDATES, ("customer_id", "account_id", "user_id"))
    if not date_column:
        date_column = guess_column(list(df.columns), DATE_CANDIDATES, ("signup_date", "join_date", "created_at", "start_date"))

    excluded = {target_column}
    if customer_id_column:
        excluded.add(customer_id_column)
    if date_column:
        excluded.add(date_column)
    # Exclude obvious identifiers even if the upload's user left the ID selector empty.
    for col in df.columns:
        key = _norm(col)
        if key in {"email", "email_address", "phone", "phone_number", "full_name", "customer_name", "address", "postcode", "zip_code"}:
            excluded.add(col)
    feature_cols = [c for c in df.columns if c not in excluded]
    if not feature_cols:
        raise ValueError("No predictor columns remain after excluding target, identifier and date columns.")

    X_all = df[feature_cols].copy()
    if text_column and text_column in X_all.columns:
        X_all[text_column] = _safe_text(X_all[text_column])
    # Remove columns which are entirely empty before fit; retain the chosen text
    # column by filling it above.
    all_empty = [c for c in X_all.columns if X_all[c].isna().all()]
    X_all = X_all.drop(columns=all_empty)
    if text_column in all_empty if text_column else False:
        text_column = None
    # Avoid OneHotEncoder failures when CSV columns contain a mixture of
    # strings and numbers (for example "Unknown" mixed into a numeric-like field).
    for col in X_all.columns:
        if col != text_column and not pd.api.types.is_numeric_dtype(X_all[col]):
            X_all[col] = X_all[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
    if not len(X_all.columns):
        raise ValueError("No usable features were found.")

    preprocessor, numeric_cols, categorical_cols, text_cols = _make_preprocessor(X_all, text_column)
    primary_pipeline = Pipeline([
        ("features", preprocessor),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=42)),
    ])
    X_labeled = X_all.loc[labeled_mask.to_numpy()]
    y_labeled = y_all_labeled
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.25, random_state=42, stratify=y_labeled,
    )
    calibrated_test = CalibratedClassifierCV(estimator=primary_pipeline, method="sigmoid", cv=3)
    calibrated_test.fit(X_train, y_train)
    test_probability = calibrated_test.predict_proba(X_test)[:, 1]
    test_metrics = _metrics(y_test.to_numpy(), test_probability)

    # A small non-calibrated Random Forest acts as a benchmark; label its raw
    # probabilities honestly in the returned model-comparison table.
    rf_preprocessor, _, _, _ = _make_preprocessor(X_all, text_column)
    rf_model = Pipeline([
        ("features", rf_preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=55, max_depth=9, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=1, random_state=42)),
    ])
    rf_model.fit(X_train, y_train)
    rf_test_probability = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = _metrics(y_test.to_numpy(), rf_test_probability)

    # Final model trains on all historical labels and scores all records. Holdout
    # metrics above remain untouched and are the estimate of unseen performance.
    final_model = CalibratedClassifierCV(estimator=primary_pipeline, method="sigmoid", cv=3)
    final_model.fit(X_labeled, y_labeled)
    all_probability = final_model.predict_proba(X_all)[:, 1]
    all_probability = np.clip(all_probability, 0.0, 1.0)

    if text_column and text_column in df.columns:
        text_values = _safe_text(df[text_column])
    else:
        text_values = pd.Series([""] * len(df), index=df.index, dtype="object")
    sentiment_scores, sentiment_labels = _sentiment_rows(text_values)
    topic_assignments, topics = _text_topics(text_values)

    segment_assignments, segment_summaries = _segments(df, X_all, encoded_y, all_probability, numeric_cols)
    medians: dict[str, float] = {}
    for col in numeric_cols:
        vals = pd.to_numeric(X_all[col], errors="coerce")
        medians[col] = float(vals.median()) if vals.notna().any() else float("nan")

    records: list[dict[str, Any]] = []
    actions_aggregate: dict[str, dict[str, Any]] = defaultdict(lambda: {"action": "", "priority_total": 0.0, "customers_flagged": 0, "risk_sum": 0.0, "reasons": Counter()})
    for i in range(len(df)):
        row = df.iloc[i]
        probability = float(all_probability[i])
        row_text = str(text_values.iloc[i])
        row_actions = _row_retention_actions(row, probability, float(sentiment_scores[i]), row_text, numeric_cols, medians)
        for action in row_actions:
            agg = actions_aggregate[action["action"]]
            agg["action"] = action["action"]
            agg["priority_total"] += float(action["priority_score"])
            agg["customers_flagged"] += 1
            agg["risk_sum"] += probability
            agg["reasons"][action["reason"]] += 1
        churn_value = None if pd.isna(encoded_y.iloc[i]) else ("Churned" if int(encoded_y.iloc[i]) == 1 else "Retained")
        customer_id = row.get(customer_id_column) if customer_id_column else None
        if customer_id is None or pd.isna(customer_id) or str(customer_id).strip() == "":
            customer_id = f"Row {i + 1}"
        records.append({
            "row_number": i + 1,
            "customer_id": str(customer_id)[:120],
            "actual_status": churn_value,
            "record_type": "Historical labelled customer" if churn_value is not None else "Unlabelled customer to score",
            "churn_probability": round(probability, 5),
            "risk_tier": _risk_tier(probability),
            "predicted_churn_at_50pct": bool(probability >= 0.5),
            "sentiment": sentiment_labels[i] if text_column else "No feedback",
            "sentiment_score": round(float(sentiment_scores[i]), 4) if text_column else None,
            "topic": topic_assignments[i],
            "segment": segment_assignments[i] if i < len(segment_assignments) else None,
            "recommended_action": row_actions[0]["action"],
            "action_reason": row_actions[0]["reason"],
            "next_best_action": row_actions[1]["action"] if len(row_actions) > 1 else None,
        })

    actions_list = []
    for item in actions_aggregate.values():
        count = max(1, int(item["customers_flagged"]))
        top_reason = item["reasons"].most_common(1)[0][0] if item["reasons"] else "Pattern-based recommendation"
        actions_list.append({
            "action": item["action"],
            "customers_flagged": int(item["customers_flagged"]),
            "average_risk": float(item["risk_sum"] / count),
            "priority_score": float(item["priority_total"] / count),
            "aggregate_priority": float(item["priority_total"]),
            "reason": top_reason,
        })
    actions_list.sort(key=lambda r: r["aggregate_priority"], reverse=True)

    y_numpy = y_test.to_numpy()
    calibration = _calibration_bins(y_numpy, test_probability)
    negative_feedback_count = sum(1 for label in sentiment_labels if label == "Negative") if text_column else 0
    labeled_count = int(labeled_mask.sum())
    churn_rate = float(y_labeled.mean())
    high_risk_count = int(np.sum(all_probability >= 0.55))
    critical_count = int(np.sum(all_probability >= 0.75))
    feedback_counts = Counter(sentiment_labels)
    feedback_summary = {
        "available": bool(text_column),
        "column": text_column,
        "counts": {key: int(value) for key, value in feedback_counts.items()},
        "negative_rate": float(negative_feedback_count / max(1, len(df))) if text_column else None,
        "average_compound": float(np.mean(sentiment_scores)) if text_column and len(sentiment_scores) else None,
        "rows_with_feedback": int(sum(t != "no feedback provided" for t in text_values.str.lower())) if text_column else 0,
    }

    # Feature importance comes from the final calibrated logistic model's
    # mean absolute coefficients across calibration folds.
    top_features = _feature_importance(final_model)
    tenure_col = _find_col(list(df.columns), ("tenure", "months_active", "customer_age_months"))
    cohorts = _cohort_analysis(df, encoded_y, date_column, tenure_col)
    model_comparison = [
        {"model": "Calibrated Logistic Regression", "calibration": "Sigmoid calibrated", **test_metrics},
        {"model": "Random Forest baseline", "calibration": "Raw probabilities (not calibrated)", **rf_metrics},
    ]
    summary = {
        "total_customers": int(len(df)),
        "labelled_customers": labeled_count,
        "unlabelled_customers": int(len(df) - labeled_count),
        "observed_churn_rate": churn_rate,
        "observed_retention_rate": 1.0 - churn_rate,
        "high_risk_customers": high_risk_count,
        "critical_risk_customers": critical_count,
        "negative_feedback_rate": feedback_summary["negative_rate"],
        "test_customers": int(len(X_test)),
        "train_customers": int(len(X_train)),
    }
    return json_safe({
        "success": True,
        "demo": False,
        "summary": summary,
        "label_info": label_info,
        "metrics": test_metrics,
        "models": model_comparison,
        "primary_model": "Calibrated Logistic Regression",
        "calibration_bins": calibration,
        "feature_importance": top_features,
        "feedback_summary": feedback_summary,
        "topics": topics,
        "actions": actions_list[:10],
        "cohorts": cohorts,
        "segments": segment_summaries,
        "predictions": records,
        "schema": {
            "target_column": target_column,
            "text_column": text_column,
            "customer_id_column": customer_id_column,
            "date_column": date_column,
            "predictor_columns": list(dict.fromkeys(numeric_cols + categorical_cols + text_cols)),
            "ignored_predictor_columns": [c for c in X_all.columns if c not in set(numeric_cols + categorical_cols + text_cols)],
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
        },
        "notes": [
            "Holdout metrics use a stratified 25% test split of labelled rows; those test rows are excluded from fitting the holdout evaluation model.",
            "Customer risk scores and recommendations use a calibrated model refit on all labelled rows, so these scores are for prioritization and are not a second unseen-test evaluation.",
            "Retention actions are heuristic recommendations, not causal estimates of incremental retention or financial uplift.",
            "NLP sentiment is a lexicon-based signal; topic extraction uses TF-IDF + NMF when there is enough varied feedback text.",
            label_info.get("mapping_note", ""),
        ],
    })


def analyze_demo() -> dict[str, Any]:
    df = _build_demo()
    # Include a small group of new, unlabelled customers so the demo shows how
    # the fitted model can score records without a known churn outcome.
    new_customer_indices = df.index[df["churn"] == 0][:12]
    df.loc[new_customer_indices, "churn"] = np.nan
    result = analyze_customers(df, "churn", "complaint_text", "customer_id", "signup_date")
    result["demo"] = True
    result["source_note"] = "Synthetic sample customers generated for demonstration; results are illustrative, not an empirical churn benchmark."
    return result
