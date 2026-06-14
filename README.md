# E-commerce IPO Analysis — Streamlit App

## Structure
```
streamlit_app/
├── app.py              # Streamlit web interface
├── api.py              # FastAPI REST API
├── requirements.txt
└── data/               # ← put your CSV files here
    ├── companies.csv
    ├── company_metrics.csv
    ├── ecommerce_index.csv
    ├── financials_annual.csv
    ├── macro_indicators.csv
    └── prices_daily.csv
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Streamlit

```bash
streamlit run app.py
```

Opens at http://localhost:8501

## Run FastAPI

```bash
uvicorn api:app --reload
```

Opens at http://localhost:8000  
Docs at http://localhost:8000/docs

## API Endpoints

| Method | URL | Description |
|---|---|---|
| GET | /companies | Filter + paginate companies |
| GET | /companies/{id} | Single company |
| GET | /stats | Summary stats by region or segment |
| POST | /companies | Add new company |

### Example GET
```
GET /companies?region=India&min_cagr=0.0&limit=10
GET /companies?segment=Fintech&limit=5&offset=0
```

### Example POST
```json
POST /companies
{
  "company_id": "myshop",
  "ticker": "MYSHP",
  "exchange": "NYSE",
  "region": "North America",
  "segment": "B2C Marketplace",
  "country_code": "US",
  "cagr": 0.12,
  "annualized_volatility": 0.45,
  "cumulative_return": 1.8
}
```

## Deploy to Streamlit Cloud

1. Push this folder to GitHub
2. Go to share.streamlit.io
3. Connect repo → select `app.py`
4. Done — you get a public URL
