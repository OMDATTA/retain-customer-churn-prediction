import numpy as np
import pandas as pd

from src.churn_engine import _build_demo, analyze_customers, encode_churn_target


def test_encode_numeric_target_with_unlabelled_rows():
    series = pd.Series([0, 1, 0, np.nan, 1])
    encoded, info = encode_churn_target(series)
    assert encoded.iloc[0] == 0
    assert encoded.iloc[1] == 1
    assert pd.isna(encoded.iloc[3])
    assert info["positive_label"] in {"1", "1.0"}


def test_encode_text_target_maps_churn_as_positive_even_if_sorted_first():
    series = pd.Series(["Churn", "Retained", "Churn", "Retained"])
    encoded, info = encode_churn_target(series)
    assert encoded.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert info["positive_label"] == "Churn"


def test_demo_analysis_reports_unseen_split_and_outputs_scores():
    data = _build_demo(180, 19)
    result = analyze_customers(data, "churn", "complaint_text", "customer_id", "signup_date")
    assert result["summary"]["test_customers"] > 0
    assert result["metrics"]["roc_auc"] is not None
    assert 0 <= result["metrics"]["brier_score"] <= 1
    assert len(result["predictions"]) == len(data)
    assert result["topics"]
    assert result["actions"]
    assert result["cohorts"]["rows"]
    assert result["segments"]


def test_unlabelled_customer_rows_are_scored_but_not_used_as_labels():
    data = _build_demo(180, 20)
    data.loc[data.index[-4:], "churn"] = np.nan
    result = analyze_customers(data, "churn", "complaint_text", "customer_id", "signup_date")
    assert result["summary"]["unlabelled_customers"] == 4
    unlabelled = [row for row in result["predictions"] if row["actual_status"] is None]
    assert len(unlabelled) == 4
    assert all(0 <= row["churn_probability"] <= 1 for row in unlabelled)
    assert all(row["record_type"] == "Unlabelled customer to score" for row in unlabelled)


def test_minimum_data_error_is_explained():
    data = _build_demo(40, 21)
    try:
        analyze_customers(data, "churn", "complaint_text", "customer_id", "signup_date")
    except ValueError as exc:
        assert "at least 12 labelled examples" in str(exc) or "at least 50 labelled" in str(exc)
    else:
        raise AssertionError("Expected data-size validation to reject a small dataset")


def test_analysis_works_without_feedback_text_column():
    data = _build_demo(150, 35).drop(columns=["complaint_text"])
    result = analyze_customers(data, "churn", None, "customer_id", "signup_date")
    assert result["feedback_summary"]["available"] is False
    assert len(result["predictions"]) == len(data)
    assert result["topics"] == []
