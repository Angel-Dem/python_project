from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import os

app = FastAPI(title="E-commerce IPO API", version="1.0")

# ── Load data once ──
DATA_DIR = "data"

def load():
    comp    = pd.read_csv(f"{DATA_DIR}/companies.csv")
    metrics = pd.read_csv(f"{DATA_DIR}/company_metrics.csv")
    df = comp.merge(metrics, on="company_id", how="inner")
    return df

try:
    df_main = load()
except Exception as e:
    df_main = pd.DataFrame()
    print(f"Warning: could not load data — {e}")


# ══════════════════════════════
# GET /companies
# ══════════════════════════════
@app.get("/companies", summary="Filter companies with pagination")
def get_companies(
    region: Optional[str] = Query(None, description="Filter by region, e.g. 'India'"),
    segment: Optional[str] = Query(None, description="Filter by segment, e.g. 'B2C Marketplace'"),
    min_cagr: Optional[float] = Query(None, description="Minimum CAGR threshold"),
    max_cagr: Optional[float] = Query(None, description="Maximum CAGR threshold"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """
    Returns filtered and paginated company records.

    - **region**: e.g. `North America`, `India`, `Europe`, `Greater China`, `Asia Pacific`, `Latin America`
    - **segment**: e.g. `B2C Marketplace`, `Fintech`, `Food Delivery`
    - **min_cagr / max_cagr**: numeric filter on annualized return
    - **limit / offset**: pagination
    """
    result = df_main.copy()

    if region:
        result = result[result["region_x"].str.lower() == region.lower()]
    if segment:
        result = result[result["segment_x"].str.lower() == segment.lower()]
    if min_cagr is not None:
        result = result[result["cagr"] >= min_cagr]
    if max_cagr is not None:
        result = result[result["cagr"] <= max_cagr]

    total = len(result)
    result = result.iloc[offset : offset + limit]

    cols = ["company_id","ticker_x","exchange","region_x","segment_x",
            "cagr","annualized_volatility","cumulative_return","country_code_x"]
    cols = [c for c in cols if c in result.columns]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": result[cols].replace({np.nan: None}).to_dict(orient="records")
    }


# ══════════════════════════════
# GET /companies/{company_id}
# ══════════════════════════════
@app.get("/companies/{company_id}", summary="Get single company by ID")
def get_company(company_id: str):
    row = df_main[df_main["company_id"] == company_id]
    if len(row) == 0:
        raise HTTPException(status_code=404, detail=f"Company '{company_id}' not found")
    return row.replace({np.nan: None}).to_dict(orient="records")[0]


# ══════════════════════════════
# POST /companies
# ══════════════════════════════
class CompanyCreate(BaseModel):
    company_id: str
    ticker: str
    exchange: str
    region: str
    segment: str
    country_code: str
    cagr: Optional[float] = None
    annualized_volatility: Optional[float] = None
    cumulative_return: Optional[float] = None

@app.post("/companies", status_code=201, summary="Add a new company")
def add_company(company: CompanyCreate):
    """
    Creates a new company record and appends it to the in-memory dataset.
    Note: changes are not persisted to disk.
    """
    global df_main

    if company.company_id in df_main["company_id"].values:
        raise HTTPException(status_code=409,
                            detail=f"Company '{company.company_id}' already exists")

    new_row = {
        "company_id": company.company_id,
        "ticker_x": company.ticker,
        "exchange": company.exchange,
        "region_x": company.region,
        "segment_x": company.segment,
        "country_code_x": company.country_code,
        "cagr": company.cagr,
        "annualized_volatility": company.annualized_volatility,
        "cumulative_return": company.cumulative_return,
    }

    df_main = pd.concat([df_main, pd.DataFrame([new_row])], ignore_index=True)

    return {"status": "created", "company_id": company.company_id}


# ══════════════════════════════
# GET /stats
# ══════════════════════════════
@app.get("/stats", summary="Summary statistics by region or segment")
def get_stats(
    group_by: str = Query("region_x", description="Column to group by: region_x or segment_x")
):
    if group_by not in ["region_x", "segment_x"]:
        raise HTTPException(status_code=400, detail="group_by must be 'region_x' or 'segment_x'")

    result = (
        df_main.groupby(group_by)[["cagr","annualized_volatility","cumulative_return"]]
        .agg(["mean","median","std","count"])
        .round(4)
    )
    result.columns = ["_".join(c) for c in result.columns]
    return result.replace({np.nan: None}).reset_index().to_dict(orient="records")
