# Global E-Commerce IPO Analysis

An interactive data analysis project examining the financial and stock-market performance of major publicly traded e-commerce companies across different countries, regions, and business segments.

The project combines company information, daily stock prices, annual financial statements, market-performance metrics, and macroeconomic indicators. The complete analytical report is available through a **Streamlit web application**, while a separate **FastAPI REST API** provides programmatic access to the company dataset.

---

## Project Overview

The analysis covers **43 publicly traded e-commerce companies** listed on major international stock exchanges, including:

* NASDAQ
* NYSE
* London Stock Exchange
* Tokyo Stock Exchange
* National Stock Exchange of India

The dataset contains companies from North America, Europe, Greater China, India, Latin America, Southeast Asia, and other regions.

The project investigates:

* long-term stock performance;
* cumulative returns and CAGR;
* annualized volatility and maximum drawdown;
* annual revenue and stock-return dynamics;
* differences between developed and emerging economies;
* relationships between financial, market, and macroeconomic indicators.

---

## Main Features

### Interactive Streamlit Report

The Streamlit application presents the full project as a structured analytical report:

1. **Abstract**
2. **Dataset Description**
3. **Descriptive Statistics**
4. **Data Cleanup**
5. **Basic Plots**
6. **Detailed Overview**
7. **Hypothesis Testing**
8. **Discussion**

The interface contains charts, statistical outputs, explanations, data-quality checks, and hypothesis-test results.

### REST API

The FastAPI service provides endpoints for:

* filtering companies;
* paginating API responses;
* retrieving a company by ID;
* calculating summary statistics;
* adding a new company record.

### Statistical Analysis

The project uses several analytical methods:

* descriptive statistics;
* correlation analysis;
* Spearman rank correlation;
* cluster bootstrap;
* permutation testing;
* Cohen’s (d);
* K-means clustering;
* PCA visualization;
* data normalization and standardization.

---

## Research Questions

### Hypothesis 1: Revenue Growth and Stock Returns

The first hypothesis examines whether annual revenue growth is positively associated with annual stock returns.

Annual company observations were constructed by combining:

* annual revenue growth from financial statements;
* annual stock returns calculated from the first and last adjusted closing prices of each year.

Because several yearly observations belong to the same company, a **cluster bootstrap by company** was used instead of treating all observations as independent.

**Result:**

* median Spearman correlation: **0.2229**;
* 95% cluster-bootstrap confidence interval: **[0.0763, 0.3732]**.

The result indicates a positive association between annual revenue growth and annual stock returns.

### Hypothesis 2: Volatility in Emerging and Developed Markets

The second hypothesis tests whether e-commerce companies from emerging economies have higher annualized stock volatility than companies from developed economies.

Countries were divided into two groups using K-means clustering based on:

* GDP per capita;
* internet usage;
* adult account ownership;
* urban population share.

The difference was evaluated using a **permutation test**, while its practical magnitude was measured using **Cohen’s (d)**.

**Result:**

| Metric                           |  Value |
| -------------------------------- | -----: |
| Emerging-market mean volatility  | 0.5076 |
| Developed-market mean volatility | 0.4733 |
| Observed difference              | 0.0343 |
| Two-sided p-value                | 0.5262 |
| One-sided p-value                | 0.2572 |
| Cohen’s (d)                      | 0.3539 |

Emerging-market companies had slightly higher average volatility, but the difference was not statistically significant.

---

## Data Sources

The project uses several related CSV tables.

| File                    | Description                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `companies.csv`         | Company descriptions, exchanges, countries, regions, segments, foundation dates, headquarters, industries, and employee counts |
| `company_metrics.csv`   | Precomputed stock-market and financial indicators for each company                                                             |
| `prices_daily.csv`      | Daily adjusted stock-price observations                                                                                        |
| `financials_annual.csv` | Annual company financial statements                                                                                            |
| `macro_indicators.csv`  | Country-level macroeconomic and digital-development indicators                                                                 |
| `ecommerce_index.csv`   | Historical global e-commerce index data                                                                                        |

Together, these tables form a composite dataset combining company-level, market-level, financial, and country-level information.

---

## Data Processing

The original tables were cleaned and transformed before analysis.

The main preprocessing steps included:

* converting date columns to `datetime`;
* correcting incomplete foundation dates;
* checking numerical columns and missing values;
* removing unusable financial observations;
* sorting stock prices by company and date;
* reconstructing missing return metrics from daily prices;
* calculating yearly stock returns;
* calculating annual revenue growth;
* forward-filling sparse macroeconomic indicators within countries;
* merging company, market, financial, and macroeconomic tables;
* standardizing macroeconomic variables before clustering.

Adjusted closing prices were used for return calculations because they account for stock splits and other historical price adjustments.

---

## Key Visualizations

The Streamlit report includes several types of visual analysis.

### Basic Exploration

* numerical distributions;
* histograms and KDE plots;
* financial-variable scatter plots;
* correlation analysis;
* descriptive-statistics tables.

### Detailed Overview

* CAGR distributions by business segment;
* equal-weighted regional stock indices;
* global e-commerce index compared with internet usage;
* annual CAGR by company and year;
* macroeconomic skewness heatmap;
* PCA visualization of developed and emerging country clusters.

### Hypothesis Testing

* annual revenue growth versus annual stock return;
* bootstrap distribution of Spearman correlations;
* comparison of volatility between developed and emerging economies;
* observed permutation-test difference.

---

## Project Structure

```text
streamlit_app/
│
├── app.py
│   └── Main Streamlit application and analytical report
│
├── api.py
│   └── FastAPI REST API
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── data/
│   ├── companies.csv
│   ├── company_metrics.csv
│   ├── ecommerce_index.csv
│   ├── financials_annual.csv
│   ├── macro_indicators.csv
│   └── prices_daily.csv
│
└── python_analysis.ipynb
    └── Data preparation, exploratory analysis, and hypothesis testing
```

> The CSV files should be stored inside the `data/` directory unless different paths are specified in `app.py` and `api.py`.

---

## Technology Stack

The project was developed using:

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **SciPy**
* **Scikit-learn**
* **Streamlit**
* **FastAPI**
* **Uvicorn**
* **Pydantic**

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

Replace `<repository-url>` and `<repository-folder>` with the address and name of your GitHub repository.

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

From the project root, run:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

The Streamlit interface contains the complete project report, including preprocessing, plots, statistical analysis, hypothesis testing, and conclusions.

---

## Running the FastAPI Service

Open a second terminal, activate the virtual environment, and run:

```bash
uvicorn api:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Alternative ReDoc documentation:

```text
http://127.0.0.1:8000/redoc
```

---

## API Endpoints

| Method | Endpoint                  | Description                                     |
| ------ | ------------------------- | ----------------------------------------------- |
| `GET`  | `/`                       | Returns general information about the API       |
| `GET`  | `/companies`              | Filters and paginates company records           |
| `GET`  | `/companies/{company_id}` | Returns one company by its identifier           |
| `GET`  | `/stats`                  | Returns summary statistics by region or segment |
| `POST` | `/companies`              | Creates a new company record                    |

---

## API Usage Examples

### Filter and paginate companies

```http
GET /companies?region=North%20America&min_cagr=0.0&limit=10&offset=0
```

```http
GET /companies?segment=Marketplace%20%2B%20Fintech&limit=5&offset=0
```

Possible query parameters include:

| Parameter  | Description                        |
| ---------- | ---------------------------------- |
| `region`   | Filter by geographical region      |
| `segment`  | Filter by business segment         |
| `min_cagr` | Minimum CAGR                       |
| `limit`    | Maximum number of returned records |
| `offset`   | Number of records to skip          |

### Retrieve one company

```http
GET /companies/amazon
```

### Request summary statistics

```http
GET /stats?group_by=region
```

```http
GET /stats?group_by=segment
```

### Add a company

```http
POST /companies
Content-Type: application/json
```

Request body:

```json
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

Example response:

```json
{
  "message": "Company created successfully",
  "company_id": "myshop"
}
```

---

## Streamlit Cloud Deployment

To publish the application online:

1. Push the complete project to GitHub.
2. Make sure `requirements.txt` contains all required dependencies.
3. Make sure the application uses relative file paths.
4. Open Streamlit Community Cloud.
5. Connect your GitHub account.
6. Select the repository and branch.
7. Set the main application file to:

```text
app.py
```

8. Start the deployment.

After deployment, Streamlit will provide a public URL for the application.

---

## Deployment Notes

For successful deployment, avoid absolute local paths such as:

```python
"C:/Users/User/Desktop/python_project/data/companies.csv"
```

Use relative paths instead:

```python
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

companies = pd.read_csv(DATA_DIR / "companies.csv")
```

This allows the same code to work locally, on GitHub, and in Streamlit Cloud.

The FastAPI application and the Streamlit application are separate services. Running or deploying Streamlit does not automatically deploy FastAPI.

---

## Reproducibility

Randomized procedures use fixed seeds where appropriate. For example:

* K-means clustering uses a fixed random seed;
* the cluster bootstrap uses a fixed random generator;
* the permutation test uses reproducible random sampling.

This makes the main statistical results reproducible across repeated executions.

---

## Limitations

The analysis should be interpreted with several limitations in mind:

* companies have different IPO and trading-history lengths;
* some macroeconomic indicators are observed only in selected years;
* financial statements are reported in different currencies;
* the number of represented countries is relatively small;
* companies from the same country or region may not be statistically independent;
* observed associations do not establish causal relationships.

---

## Repository Purpose

This repository was created as a university data-analysis project.

Its purpose is to demonstrate:

* work with a composite dataset;
* data cleaning and transformation;
* exploratory data analysis;
* statistical hypothesis testing;
* interactive data visualization;
* REST API development;
* web deployment with Streamlit.

---

## Author

**Angelina**
HSE University — Applied Data Analysis
