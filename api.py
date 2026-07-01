from pathlib import Path
from shutil import copy2
from threading import Lock
import csv
import json

import pandas as pd

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import RedirectResponse


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent


SOURCE_DATA_PATH = BASE_DIR / "financials_annual.csv"

API_DATA_PATH = BASE_DIR / "financials_api.csv"

WRITE_LOCK = Lock()


def prepare_storage() -> None:
    """
    Creates a working copy of the dataset on the first API launch.
    """
    if not SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {SOURCE_DATA_PATH}"
        )

    if not API_DATA_PATH.exists():
        copy2(SOURCE_DATA_PATH, API_DATA_PATH)


def load_financials() -> pd.DataFrame:
    prepare_storage()
    return pd.read_csv(API_DATA_PATH)


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """
    Converts pandas DataFrame to JSON-compatible records.
    NaN values become null.
    """
    return json.loads(
        df.to_json(
            orient="records"
        )
    )


class FinancialRecordCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(
        min_length=1,
        max_length=100
    )

    ticker: str = Field(
        min_length=1,
        max_length=20
    )

    fiscal_year: int = Field(
        ge=1900,
        le=2100
    )

    total_revenue: float = Field(
        ge=0
    )

    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    ebitda: float | None = None

    research_dev: float | None = Field(
        default=None,
        ge=0
    )

    total_assets: float | None = Field(
        default=None,
        ge=0
    )

    total_liabilities: float | None = Field(
        default=None,
        ge=0
    )

    cash_and_equivalents: float | None = Field(
        default=None,
        ge=0
    )

    total_equity: float | None = None
    operating_cash_flow: float | None = None
    capex: float | None = None
    free_cash_flow: float | None = None



app = FastAPI(
    title="Global E-commerce Financials API",
    description=(
        "REST API for obtaining and adding annual financial "
        "records of global e-commerce companies."
    ),
    version="1.0.0"
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")



@app.get("/financials")
def get_financials(
    ticker: str | None = Query(
        default=None,
        min_length=1,
        description="Company ticker, for example AMZN"
    ),

    year_from: int | None = Query(
        default=None,
        ge=1900,
        le=2100,
        description="First fiscal year"
    ),

    year_to: int | None = Query(
        default=None,
        ge=1900,
        le=2100,
        description="Last fiscal year"
    ),

    offset: int = Query(
        default=0,
        ge=0,
        description="Number of records to skip"
    ),

    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of returned records"
    )
):
    """
    Returns annual financial records.

    Supports:
    - filtering by ticker;
    - filtering by fiscal-year interval;
    - pagination through offset and limit.
    """

    if (
        year_from is not None
        and year_to is not None
        and year_from > year_to
    ):
        raise HTTPException(
            status_code=400,
            detail="year_from must not exceed year_to"
        )

    df = load_financials()

    if ticker:
        normalized_ticker = ticker.strip().upper()

        df = df[
            df["ticker"]
            .astype(str)
            .str.upper()
            .eq(normalized_ticker)
        ]

    if year_from is not None:
        fiscal_year = pd.to_numeric(
            df["fiscal_year"],
            errors="coerce"
        )

        df = df[
            fiscal_year >= year_from
        ]

    if year_to is not None:
        fiscal_year = pd.to_numeric(
            df["fiscal_year"],
            errors="coerce"
        )

        df = df[
            fiscal_year <= year_to
        ]

    df = df.sort_values(
        ["ticker", "fiscal_year"]
    )

    total = len(df)

    page = df.iloc[
        offset:offset + limit
    ]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": dataframe_to_records(page)
    }



@app.post(
    "/financials",
    status_code=status.HTTP_201_CREATED
)
def create_financial_record(
    record: FinancialRecordCreate
):
    """
    Creates a new annual financial record.
    """

    with WRITE_LOCK:
        df = load_financials()

        company_id = (
            record.company_id
            .strip()
            .lower()
        )

        ticker = (
            record.ticker
            .strip()
            .upper()
        )

        fiscal_year = pd.to_numeric(
            df["fiscal_year"],
            errors="coerce"
        )

        duplicate = (
            df["company_id"]
            .astype(str)
            .str.lower()
            .eq(company_id)
            &
            fiscal_year.eq(record.fiscal_year)
        ).any()

        if duplicate:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A financial record for this company "
                    "and fiscal year already exists"
                )
            )


        new_row = {
            column: None
            for column in df.columns
        }

        new_row.update(
            record.model_dump(
                exclude_none=True
            )
        )

        new_row["company_id"] = company_id
        new_row["ticker"] = ticker

        with API_DATA_PATH.open(
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=df.columns
            )

            writer.writerow(new_row)

    return {
        "message": "Financial record created successfully",
        "item": new_row
    }