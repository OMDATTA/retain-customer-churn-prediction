"""Vercel-ready API for the Churn & Feedback Intelligence project.

Vercel serves files from public/ through its CDN. This FastAPI app only handles
API routes; static mounting is enabled solely for local development.
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.churn_engine import (
    MAX_COLUMNS,
    MAX_ROWS,
    analyze_customers,
    analyze_demo,
    guess_column,
    preview_frame,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_feedback_studio")
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
MAX_CSV_BYTES = 3 * 1024 * 1024  # Keep below Vercel's 4.5 MB function body limit.

app = FastAPI(
    title="Churn & Feedback Intelligence API",
    description=(
        "Calibrated customer-churn prediction, complaint sentiment and topic mining, "
        "retention-action ranking, cohort analysis, and customer segmentation."
    ),
    version="1.0.0",
)


def _read_csv_bytes(contents: bytes, *, preview: bool = False) -> pd.DataFrame:
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    if len(contents) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail="CSV exceeds this Vercel-friendly build's 3 MiB upload limit. Reduce the file or run larger analyses in a notebook/hosted data service.",
        )
    nrows = MAX_ROWS + 1
    try:
        frame = pd.read_csv(io.BytesIO(contents), nrows=nrows, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            frame = pd.read_csv(io.BytesIO(contents), nrows=nrows, low_memory=False, encoding="latin-1")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse this CSV: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse this CSV: {exc}") from exc
    if frame.empty:
        raise HTTPException(status_code=400, detail="The CSV has a header but no data rows.")
    if len(frame.columns) > MAX_COLUMNS:
        raise HTTPException(status_code=422, detail=f"CSV has {len(frame.columns)} columns; please reduce it to {MAX_COLUMNS} or fewer.")
    if not preview and len(frame) > MAX_ROWS:
        raise HTTPException(status_code=422, detail=f"CSV has more than {MAX_ROWS:,} rows. This Vercel-friendly build caps an analysis at {MAX_ROWS:,} rows to keep model training within function limits.")
    return frame


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    # On Vercel, public/index.html is served directly by the CDN before this route.
    # This redirect is a helpful fallback for local/alternate deployments.
    return RedirectResponse(url="/index.html", status_code=307)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "churn-feedback-studio", "version": "1.0.0"}


@app.get("/api/info")
def info() -> dict:
    return {
        "service": "Churn & Feedback Intelligence",
        "features": [
            "sigmoid-calibrated churn probabilities",
            "stratified holdout evaluation on unseen labelled rows",
            "Random Forest baseline comparison",
            "complaint sentiment and TF-IDF/NMF topics",
            "retention-action ranking",
            "cohort analysis and KMeans segments",
            "scores for rows whose churn labels are blank",
        ],
        "limits": {"max_upload_bytes": MAX_CSV_BYTES, "max_rows": MAX_ROWS, "max_columns": MAX_COLUMNS},
        "storage": "Uploads are processed per request; this prototype does not persist uploaded files to a database.",
    }


@app.post("/api/preview")
async def preview_csv(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a .csv file. Excel workbooks are not supported by this endpoint.")
    contents = await file.read(MAX_CSV_BYTES + 1)
    frame = _read_csv_bytes(contents, preview=True)
    result = preview_frame(frame)
    result["row_limit_exceeded"] = len(frame) > MAX_ROWS
    if len(frame) > MAX_ROWS:
        result["note"] = f"The file has at least {MAX_ROWS + 1:,} rows. Reduce it to {MAX_ROWS:,} rows or fewer before analysis."
    return result


@app.post("/api/analyze")
async def analyze_csv(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    text_column: Optional[str] = Form(default=None),
    customer_id_column: Optional[str] = Form(default=None),
    date_column: Optional[str] = Form(default=None),
) -> dict:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a .csv file.")
    contents = await file.read(MAX_CSV_BYTES + 1)
    frame = _read_csv_bytes(contents, preview=False)
    try:
        result = analyze_customers(
            frame,
            target_column=target_column.strip(),
            text_column=(text_column or "").strip() or None,
            customer_id_column=(customer_id_column or "").strip() or None,
            date_column=(date_column or "").strip() or None,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Customer churn analysis failed")
        raise HTTPException(status_code=500, detail="Analysis failed unexpectedly. Check that the CSV has valid binary churn labels and enough labelled examples; see Vercel Runtime Logs if the problem persists.") from exc


@app.post("/api/demo")
def demo_analysis() -> dict:
    """Run a repeatable synthetic demo; it does not use or store customer data."""
    try:
        return analyze_demo()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Demo analysis failed")
        raise HTTPException(status_code=500, detail="The demo analysis failed unexpectedly. Check Runtime Logs for the traceback.") from exc


# Vercel's CDN serves public/** as static assets. Do not mount StaticFiles on
# Vercel; doing so can shadow the API and/or produce missing-asset errors.
if os.getenv("VERCEL") != "1" and PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
