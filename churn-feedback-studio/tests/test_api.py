from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_info_has_upload_limits():
    response = client.get("/api/info")
    assert response.status_code == 200
    assert response.json()["limits"]["max_rows"] == 2500


def test_csv_preview_detects_expected_columns():
    response = client.post(
        "/api/preview",
        files={"file": ("customers.csv", b"customer_id,churn,complaint_text,signup_date\nA,0,Good service,2025-01-01\nB,1,Bad service,2025-01-02\n", "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["target_suggestion"] == "churn"
    assert data["text_suggestion"] == "complaint_text"
    assert data["customer_id_suggestion"] == "customer_id"
    assert data["date_suggestion"] == "signup_date"


def test_analyze_endpoint_validates_small_target_data():
    csv_bytes = b"customer_id,churn,value\n" + b"".join(
        f"C{i},{1 if i % 5 == 0 else 0},{i % 13}\n".encode() for i in range(24)
    )
    response = client.post(
        "/api/analyze",
        files={"file": ("small.csv", csv_bytes, "text/csv")},
        data={"target_column": "churn", "customer_id_column": "customer_id"},
    )
    assert response.status_code == 422
    assert "at least" in response.json()["detail"]


def test_demo_endpoint_returns_full_dashboard_payload():
    response = client.post("/api/demo")
    assert response.status_code == 200
    data = response.json()
    assert data["demo"] is True
    assert data["summary"]["total_customers"] > 100
    assert "roc_auc" in data["metrics"]
    assert data["predictions"]


def test_sample_csv_can_be_analyzed_end_to_end():
    from pathlib import Path

    sample_path = Path(__file__).resolve().parents[1] / "public" / "sample-churn-data.csv"
    response = client.post(
        "/api/analyze",
        files={"file": ("sample-churn-data.csv", sample_path.read_bytes(), "text/csv")},
        data={"target_column": "churn", "text_column": "complaint_text", "customer_id_column": "customer_id", "date_column": "signup_date"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["summary"]["unlabelled_customers"] >= 1
    assert payload["summary"]["total_customers"] == 260
    assert len(payload["predictions"]) == 260
