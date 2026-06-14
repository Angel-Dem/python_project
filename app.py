import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs_module
import seaborn as sns
from scipy import stats
from scipy.cluster.vq import kmeans, vq
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="E-commerce IPO Analysis", layout="wide", page_icon="📈")

# ──────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────

@st.cache_data
def load_data():
    df_comp     = pd.read_csv("companies.csv")
    df_metrics  = pd.read_csv("company_metrics.csv")
    df_ecommerce= pd.read_csv("ecommerce_index.csv")
    df_fin      = pd.read_csv("financials_annual.csv")
    df_macro    = pd.read_csv("macro_indicators.csv")
    df_stick    = pd.read_csv("prices_daily.csv")
    return df_comp, df_metrics, df_ecommerce, df_fin, df_macro, df_stick

@st.cache_data
def fill_missing_stock_metrics(df_filled, df_stock):
    df_filled = df_filled.copy()
    df_stock = df_stock.copy()

    df_stock["date"] = pd.to_datetime(df_stock["date"])

    for idx in df_filled.index:
        if pd.notna(df_filled.loc[idx, "cumulative_return"]):
            continue

        cid = df_filled.loc[idx, "company_id"]

        tmp = (
            df_stock[df_stock["company_id"] == cid]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(tmp) < 2:
            continue

        first_price = tmp.iloc[0]["close"]
        last_price = tmp.iloc[-2]["close"]
        last_date = tmp.iloc[-2]["date"]

        df_filled.loc[idx, "last_close"] = last_price
        df_filled.loc[idx, "cumulative_return"] = last_price / first_price - 1

        hs = pd.to_datetime(df_filled.loc[idx, "history_start"])
        he = pd.to_datetime(df_filled.loc[idx, "history_end"])
        years = (he - hs).days / 365.25

        if years > 0:
            df_filled.loc[idx, "cagr"] = (last_price / first_price) ** (1 / years) - 1

        adj_last = tmp.iloc[-2]["adj_close"]

        target_1y = last_date - pd.DateOffset(years=1)
        s1y = tmp.loc[tmp["date"] <= target_1y, "adj_close"]

        if len(s1y) > 0:
            df_filled.loc[idx, "return_1y"] = adj_last / s1y.iloc[-1] - 1

        target_90d = last_date - pd.DateOffset(days=128)
        s90 = tmp.loc[tmp["date"] <= target_90d, "adj_close"]

        if len(s90) > 0:
            df_filled.loc[idx, "return_90d"] = adj_last / s90.iloc[-1] - 1

    return df_filled

def prepare_company_data(df_comp, df_metrics, df_stock):
    df = df_comp.merge(df_metrics, on="company_id", how="inner")

    cols_to_drop = [
        "headquarters", "founders", "employees", "employees_year",
        "name", "reporting_currency", "wikipedia_title", "wikidata_qid",
        "wiki_title_canonical", "website", "wiki_extract", "wiki_url",
        "wiki_thumbnail"
    ]

    df_filled = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    fixes = {
        "carvana": "2012-01-04",
        "sea": "2009-05-08",
        "delhivery": "2011-05-01",
        "affirm": "2012-01-17"
    }

    for cid, date in fixes.items():
        df_filled.loc[df_filled["company_id"] == cid, "founded_date"] = date

    df_filled["founded_date"] = (
        df_filled["founded_date"]
        .str.replace("-00-", "-01-", regex=False)
        .str.replace(r"-00$", "-01", regex=True)
    )

    df_filled["founded_date"] = pd.to_datetime(
        df_filled["founded_date"],
        errors="coerce"
    )

    df_filled = fill_missing_stock_metrics(df_filled, df_stock)

    return df_filled

def prepare_financial_data(df_fin):
    df_fin_full = df_fin.copy()

    critical_cols = [
        "net_income", "operating_income", "total_revenue", "ebitda",
        "total_assets", "total_liabilities", "cash_and_equivalents",
        "total_equity", "capex", "free_cash_flow"
    ]

    mask = df_fin_full[critical_cols].isna().all(axis=1)
    df_fin_full = df_fin_full.loc[~mask].copy()

    df_fin_full["operating_margin"] = (
        df_fin_full["operating_income"] / df_fin_full["total_revenue"]
    )

    df_fin_full["rd_intensity"] = (
        df_fin_full["research_dev"] / df_fin_full["total_revenue"]
    )

    df_fin_full["has_rd"] = df_fin_full["research_dev"].notna().astype(int)

    goto_mask = df_fin_full["ticker"] == "GOTO.JK"

    df_fin_full.loc[goto_mask, "operating_cash_flow"] = (
        df_fin_full.loc[goto_mask, "free_cash_flow"]
        - df_fin_full.loc[goto_mask, "capex"]
    )

    return df_fin_full

def prepare_macro_data(df_macro):
    macro_wide = (
        df_macro
        .pivot_table(
            index=["country_code", "year"],
            columns="indicator_name",
            values="value"
        )
        .reset_index()
        .sort_values(["country_code", "year"])
    )

    macro_wide["account_ownership_pct_adult"] = (
        macro_wide
        .groupby("country_code")["account_ownership_pct_adult"]
        .ffill()
        .bfill()
    )

    m_mask = macro_wide["gdp_total_usd"].isna()

    macro_wide.loc[m_mask, "gdp_total_usd"] = (
        macro_wide.loc[m_mask, "gdp_per_capita_usd"]
        * macro_wide.loc[m_mask, "population_total"]
    )

    cols_to_fill = [
        "gdp_per_capita_usd", "internet_users_pct", "fdi_inflow_pct_gdp",
        "mobile_subs_per_100", "gdp_total_usd", "population_total",
        "urban_population_pct"
    ]

    for col in cols_to_fill:
        macro_wide[col] = (
            macro_wide
            .groupby("country_code")[col]
            .ffill()
            .bfill()
        )

    return macro_wide

def create_country_clusters(macro_wide):
    features = [
        "gdp_per_capita_usd",
        "internet_users_pct",
        "account_ownership_pct_adult",
        "urban_population_pct"
    ]

    country_df = (
        macro_wide
        .groupby("country_code")[features]
        .mean()
        .reset_index()
    )

    X = country_df[features].values
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    centroids, _ = kmeans(X_scaled, 2, seed=42)
    labels, _ = vq(X_scaled, centroids)

    country_df["cluster"] = labels

    dev_cluster = (
        country_df
        .groupby("cluster")["gdp_per_capita_usd"]
        .mean()
        .idxmax()
    )

    country_df["market_type"] = country_df["cluster"].apply(
        lambda x: "Developed" if x == dev_cluster else "Emerging"
    )

    X_centered = X_scaled - X_scaled.mean(axis=0)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    coords = X_centered @ Vt.T[:, :2]

    country_df["pca1"] = coords[:, 0]
    country_df["pca2"] = coords[:, 1]

    return country_df

def create_company_analysis_dataset(df_filled, df_fin2, macro_wide, country_df):
    macro_latest = (
        macro_wide
        .sort_values("year")
        .groupby("country_code")
        .last()
        .reset_index()
    )[[
        "country_code",
        "internet_users_pct",
        "gdp_per_capita_usd",
        "fdi_inflow_pct_gdp"
    ]]

    df_fin_latest = (
        df_fin2
        .sort_values("fiscal_year")
        .groupby("company_id")
        .last()
        .reset_index()
    )[[
        "company_id",
        "operating_margin",
        "free_cash_flow",
        "total_revenue",
        "total_assets",
        "ebitda",
        "rd_intensity",
        "has_rd",
        "operating_cash_flow"
    ]]

    df_companies = df_filled.drop_duplicates("company_id").copy()

    df_companies = df_companies.merge(
        df_fin_latest,
        on="company_id",
        how="left"
    )

    df_companies = df_companies.merge(
        macro_latest,
        left_on="country_code_x",
        right_on="country_code",
        how="left"
    )

    if "country_code" in df_companies.columns:
        df_companies = df_companies.drop(columns=["country_code"])

    df_companies = df_companies.merge(
        country_df[["country_code", "market_type", "cluster"]],
        left_on="country_code_x",
        right_on="country_code",
        how="left"
    )

    return df_companies

def prepare_stock_data(df_stock, df_filled, country_df):
    df_stock_copy = df_stock.copy()

    df_stock_copy["date"] = pd.to_datetime(df_stock_copy["date"])

    df_stock_copy = df_stock_copy.merge(
        df_filled[["company_id", "country_code_x"]].drop_duplicates(),
        on="company_id",
        how="left"
    )

    df_stock_copy = df_stock_copy.rename(
        columns={"country_code_x": "country_code"}
    )

    df_stock_copy = df_stock_copy.merge(
        country_df[["country_code", "market_type"]],
        on="country_code",
        how="left"
    )

    df_stock_copy = df_stock_copy.sort_values(["company_id", "date"])

    df_stock_copy["daily_return"] = (
        df_stock_copy
        .groupby("company_id")["adj_close"]
        .pct_change(fill_method=None)
    )

    df_stock_copy["rolling_vol"] = (
        df_stock_copy
        .groupby("company_id")["daily_return"]
        .transform(lambda x: x.rolling(252).std() * np.sqrt(252))
    )

    df_stock_copy["year"] = df_stock_copy["date"].dt.year

    return df_stock_copy

def find_outliers(series):
    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[(series < lower) | (series > upper)]

    return outliers, lower, upper


def outlier_scan(dataframe):
    numeric_cols = dataframe.select_dtypes(include=np.number).columns

    rows = []

    for col in numeric_cols:
        outliers, lower, upper = find_outliers(dataframe[col])

        rows.append({
            "column": col,
            "outlier_count": len(outliers),
            "outlier_share": len(outliers) / dataframe[col].dropna().shape[0] if dataframe[col].dropna().shape[0] > 0 else np.nan,
            "lower_bound": lower,
            "upper_bound": upper
        })

    return (
        pd.DataFrame(rows)
        .sort_values("outlier_count", ascending=False)
        .reset_index(drop=True)
    )


def quick_eda(dataframe, name="Dataset"):
    numeric_cols = dataframe.select_dtypes(include=np.number).columns
    categorical_cols = dataframe.select_dtypes(exclude=np.number).columns

    total_missing = dataframe.isna().sum().sum()

    missing_top = (
        dataframe
        .isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    missing_top.columns = ["column", "missing_count"]
    missing_top["missing_share"] = (
        missing_top["missing_count"] / len(dataframe)
    ).round(3)

    skew_top = (
        dataframe[numeric_cols]
        .skew()
        .abs()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    skew_top.columns = ["column", "abs_skewness"]

    corr_pairs = []

    if len(numeric_cols) >= 2:
        corr = dataframe[numeric_cols].corr().abs()

        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        corr_pairs = (
            upper
            .stack()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        corr_pairs.columns = ["column_1", "column_2", "abs_correlation"]
    else:
        corr_pairs = pd.DataFrame(
            columns=["column_1", "column_2", "abs_correlation"]
        )

    summary = {
        "dataset": name,
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "total_missing_values": int(total_missing)
    }

    return summary, missing_top, skew_top, corr_pairs


def cohen_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.var(ddof=1)+(n2-1)*g2.var(ddof=1))/(n1+n2-2))
    return (g1.mean()-g2.mean())/pooled


# ──────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────

df_comp, df_metrics, df_ecommerce, df_fin_raw, df_macro, df_stick = load_data()
df_filled = prepare_company_data(df_comp, df_metrics, df_stick)
df_fin = prepare_financial_data(df_fin_raw)
macro_wide = prepare_macro_data(df_macro)
country_df = create_country_clusters(macro_wide)
df_companies = create_company_analysis_dataset(df_filled, df_fin, macro_wide, country_df)
df_stock_copy = prepare_stock_data(df_stick, df_filled, country_df)
df = df_comp.merge(df_metrics, on="company_id", how="inner")

# ──────────────────────────────────────────
# NAVIGATION
# ──────────────────────────────────────────

st.sidebar.title("📈 E-commerce IPO Analysis")
page = st.sidebar.radio("Navigation", [
    "Abstract",
    "Dataset Description",
    "Descriptive Statistics",
    "Data Cleanup",
    "Basic Plots",
    "Detailed Overview",
    "Hypothesis Testing",
    "Discussion"
])

# ══════════════════════════════════════════
# 1. ABSTRACT
# ══════════════════════════════════════════

if page == "Abstract":
    st.title("E-commerce IPO Performance: Macroeconomic and Regional Drivers")
    st.markdown("""
    ### Abstract
This project investigates the post-IPO stock performance of e-commerce companies across different 
regions and examines how country-level macroeconomic conditions may be associated with their
 market outcomes. The analysis is based on a composite dataset of 43 publicly 
traded e-commerce companies listed on multiple stock exchanges, 
including the NYSE, NASDAQ, LSE, NSE, TSE, and others. 
It als combines annual company financial data for 2015–2024, 
daily stock prices from 2015 to these days, and country macroeconomic indicators. 
This structure make it possible to check relationships between different types of data, 
investigate the nature of missing values, and test 2 central hypotheses.

**H1** examines whether macroeconomic conditions—specifically internet penetration, GDP per capita, and foreign direct investment inflows—are associated with companies’ long-term post-IPO stock performance, measured by the compound annual growth rate (CAGR).
**H2** investigates whether e-commerce companies from emerging markets exhibit systematically higher post-IPO stock-price volatility than companies from developed markets.

The analysis pipeline includes the reconstruction of missing market-performance 
indicators using daily stock-price data, verification of the relevant financial formulas against the reported values,
 and K-Means clustering of countries into developed- and emerging-market groups. 
The hypotheses are evaluated using correlation analysis, company-level bootstrap confidence intervals,
 permutation tests and Cohen’s estimation

    The work was done individually
    ---
    *Dataset source: [Kaggle — E-commerce Dataset](https://www.kaggle.com/datasets/parampratap/ecommerce-dataset)*  
    *Author: individual project*
    """)

# ══════════════════════════════════════════
# 2. DATASET DESCRIPTION
# ══════════════════════════════════════════

elif page == "Dataset Description":
    st.title("Dataset Description")

    st.markdown("""
    The dataset belongs to the **financial and investment domain**. 
    The main goal is to analyze post-IPO performance of global e-commerce companies using company-level,
    financial, stock-market and macroeconomic data.

    This is a **composite dataset**, not a single CSV file. It combines company metadata, stock performance
    metrics, annual financial statements, daily prices, macroeconomic indicators and an e-commerce benchmark index.
    """)

    st.subheader("Source tables")

    dataset_summary = pd.DataFrame({
        "Dataset": [
            "companies.csv",
            "company_metrics.csv",
            "financials_annual.csv",
            "macro_indicators.csv",
            "prices_daily.csv",
            "ecommerce_index.csv"
        ],
        "Rows": [
            len(df_comp),
            len(df_metrics),
            len(df_fin_raw),
            len(df_macro),
            len(df_stick),
            len(df_ecommerce)
        ],
        "Columns": [
            df_comp.shape[1],
            df_metrics.shape[1],
            df_fin_raw.shape[1],
            df_macro.shape[1],
            df_stick.shape[1],
            df_ecommerce.shape[1]
        ],
        "Main content": [
            "Company identifiers, tickers, country, region, segment, foundation data and descriptive fields",
            "Latest market-performance and financial metrics without yearly structure",
            "Annual financial statements by company and fiscal year",
            "Country-level macroeconomic indicators by year",
            "Daily stock prices and trading data",
            "Benchmark global e-commerce index over time"
        ]
    })

    st.dataframe(dataset_summary, use_container_width=True)

    st.markdown("""
    The two tables `companies.csv` and `company_metrics.csv` contain **one row per company**.
    They describe the latest available company-level information and do not have a separate date column
    for each observation. Therefore, I merge them into one company-level dataframe using `company_id`.
    """)

    df_company_level = df_comp.merge(
        df_metrics,
        on="company_id",
        how="inner"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Companies in companies.csv", df_comp["company_id"].nunique())
    col2.metric("Companies in company_metrics.csv", df_metrics["company_id"].nunique())
    col3.metric("Companies after merge", df_company_level["company_id"].nunique())

    st.subheader("Company IDs in companies.csv")
    st.write(sorted(df_comp["company_id"].unique()))

    st.subheader("Company IDs in company_metrics.csv")
    st.write(sorted(df_metrics["company_id"].unique()))

    st.subheader("Merged company-level dataset")
    st.write(f"Shape after merge: **{df_company_level.shape[0]} rows × {df_company_level.shape[1]} columns**")
    st.dataframe(df_company_level.head(5), use_container_width=True)

    st.markdown("""
    After merging, columns with the same names from both tables receive suffixes such as `_x` and `_y`.
    For example, `ticker_x`, `region_x` and `segment_x` come from `companies.csv`, while the corresponding
    `_y` columns come from `company_metrics.csv`. In the following analysis, I will delete unimportant and repetitive data
    that were mantioned(as redion_x and region_y as they represent the same information), also data containig url to company
    description will be also deleted. Moreover, below the new merged dataset from company description and company metrices is the only one
    to use
    """)

    st.subheader("Companies by region and segment")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Companies by region**")
        st.bar_chart(df["region_x"].value_counts())

    with col2:
        st.write("**Companies by segment**")
        st.bar_chart(df["segment_x"].value_counts())

    st.subheader("Data types by dataset")

    datasets = {
        "companies.csv": df_comp,
        "company_metrics.csv": df_metrics,
        "financials_annual.csv": df_fin_raw,
        "macro_indicators.csv": df_macro,
        "prices_daily.csv": df_stick,
        "ecommerce_index.csv": df_ecommerce,
        "merged new dataset": df
    }

    selected_dtype_dataset = st.selectbox(
        "Choose dataset to inspect data types",
        list(datasets.keys())
    )

    dtype_table = (
        datasets[selected_dtype_dataset]
        .dtypes
        .astype(str)
        .rename("dtype")
        .to_frame()
    )

    st.dataframe(dtype_table, use_container_width=True)

    st.markdown("""
    The dataset contains several field types:

    - **Numerical fields:** revenue, net income, total assets, free cash flow, CAGR, cumulative return,
      volatility, stock prices and macroeconomic indicators.
    - **Categorical fields:** company ID, ticker, exchange, country, region and business segment.
    - **Temporal fields:** fiscal year, daily stock date, history start/end dates and index dates.
    """)

    st.subheader("Missing values by dataset")

    selected_missing_dataset = st.selectbox(
        "Choose dataset to inspect missing values",
        list(datasets.keys()),
        key="missing_dataset_selector"
    )

    missing_table = (
        datasets[selected_missing_dataset]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .rename("missing_count")
        .to_frame()
    )

    missing_table["missing_share"] = (
        missing_table["missing_count"] / len(datasets[selected_missing_dataset])
    ).round(3)

    st.dataframe(missing_table, use_container_width=True)

    st.markdown("""
    Several data quality issues were identified:

    1. `company_metrics.csv` has missing values in stock-return metrics such as `cagr`,
       `last_close`, `cumulative_return`, `return_1y` and `return_90d`.
       These values will not be dropped immediately. Instead, I reconstruct them from
       `prices_daily.csv` using the original price-based formulas.

    2. `companies.csv` contains descriptive columns with many missing values or low analytical value,
       such as `employees`, `employees_year`, `founders`, `wiki_thumbnail`, `website`, `wiki_url`
       and long Wikipedia text fields. These columns are removed before the main analysis in merged dataset.

    3. `financials_annual.csv` contains missing values in financial indicators.
       In some cases, the first fiscal year of reporting has almost no financial data.
       Such rows are treated as incomplete reporting periods and are removed if all critical
       financial fields are missing.

    4. `research_dev` has many missing values. This does not necessarily mean zero R&D spending:
       some firms may not report this item separately. Therefore, I create an additional binary
       indicator `has_rd`, showing whether R&D expenses are reported.

    5. `operating_cash_flow` has several missing values. For GoTo Group, it can be reconstructed
       using the accounting relation:

       `operating_cash_flow = free_cash_flow - capex`

    6. `macro_indicators.csv` is structurally clean, but some indicators are not reported every year.
       For example, account ownership is surveyed only in selected years. These gaps are filled within
       each country using forward-fill and backward-fill.

    Overall, the dataset is suitable for analysis, but it requires preprocessing before hypothesis testing.
    The most important cleaning steps are formula-based reconstruction of missing return metrics,
    removal of uninformative descriptive columns, financial statement cleanup and macroeconomic
    imputation by country.
    """)

    st.subheader("Sample rows after initial cleaning")
    st.dataframe(df_filled.head(5), use_container_width=True)

# ══════════════════════════════════════════
# 3. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════

elif page == "Descriptive Statistics":
    st.title("Descriptive Statistics")

    st.markdown("""
    in this section descriptive statistics for the raw datasets before the main cleaning stage are showen.
    I represented the statistics based on three datasets: the merged company-level dataset, annual financial statements, 
    and macroeconomic indicators.
    """)


    datasets_for_eda = {
        "Merged companies + metrics": df,
        "Annual financials": df_fin_raw,
        "Macro indicators": df_macro
    }

    # ─────────────────────────────
    # QUICK EDA
    # ─────────────────────────────

    st.subheader("1. Quick EDA report")

    selected_dataset = st.selectbox(
        "Choose dataset for quick EDA",
        list(datasets_for_eda.keys())
    )

    selected_df = datasets_for_eda[selected_dataset]

    summary, missing_top, skew_top, corr_pairs = quick_eda(
        selected_df,
        selected_dataset
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", summary["rows"])
    col2.metric("Columns", summary["columns"])
    col3.metric("Numeric columns", summary["numeric_columns"])
    col4.metric("Missing values", summary["total_missing_values"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Top columns with missing values**")
        st.dataframe(missing_top, use_container_width=True)

    with col2:
        st.write("**Top most skewed numeric columns**")
        st.dataframe(skew_top, use_container_width=True)

    with col3:
        st.write("**Top correlated numeric pairs**")
        st.dataframe(corr_pairs, use_container_width=True)

    if selected_dataset == "Merged companies + metrics":
        st.caption("""Missed values""")
        st.markdown("""
For the dataset obtained by merging company information with company-level metrics, Ш identified a relatively large number of missing values. The complete summary of missing values by column is presented on the previous page.

The most important result of this analysis is that several columns provide little analytical value and/or contain too many missing observations. Therefore, the following columns will be removed from the main dataset:

- headquarters
- founders
- employees
- employees_year
- name
- reporting_currency
- wikipedia_title
- wikidata_qid
- wiki_title_canonical
- website
- wiki_extract
- wiki_url
- wiki_thumbnail

The remaining missing values are concentrated mainly in:

- return_90d
- return_1y
- cumulative_return
- last_close
- cagr

These variables will not be removed. Moreover, in the next chapter they will be reconstructed 
using the original financial formulas.

The reconstruction process consists of two steps:

1. Deriving the formulas from the available stock-price data.
2. Comparing the calculated values with the reported values for companies where the metrics are available.

If the difference between the calculated and reported values is close to zero, the formula can be considered verified and used to fill the missing observations.
Moreover,
                    1. Since there are only 4 out of 43 passes for founded_date, the dates will be checked manually
                    2. to fill in last_close, we will look at the last price in stock performance
                    
                    """)
        st.caption("""Skewness""")
        st.markdown("""
Financial scale variables exhibit substantial positive skewness. 
This indicates that a small number of firms account for a disproportionately large share of revenue, 
profits and cash generation. Such behaviour is expected in e-commerce markets where a few dominant firms 
coexist with many smaller companies as in the case of this collected data.
        """)
        st.caption("""Correlation betwen values""")
        st.markdown("""


The results show rather high correlations, especially between financial variables such as 
`total_revenue`, `net_income`, `operating_income`, `free_cash_flow`, and `total_assets`.

The strongest correlation is observed between `total_revenue` and `net_income` with an absolute correlation 
of **0.975**. This result is not actually that surprising, 
because these variables are directly connected to the scale of a company.
Larger companies usually generate higher revenue, and this often leads to larger absolute values of 
profit or loss. Therefore, this correlation mostly depends on company size rather than proving that 
revenue alone explains profitability.

Similarly, `net_income` and `operating_income` have a strong correlation of **0.941**. 
This is also expected because both indicators show different levels of profit: 
operating income depends on profit from core business operations, 
while net income is the final profit after taxes, interest, and other non-operating items. 
Since they are based on related accounting concepts, their strong relationship is natural.

High correlations between `operating_income`, `net_income`, and `free_cash_flow` 
also suggest that companies with stronger operating performance tend to have 
stronger cash flow results. However, this relationship should be interpreted carefully 
because free cash flow is affected not only by profitability, 
but also by capital expenditures, working capital changes, and company-specific investment decisions.

The correlation between `total_revenue` and `total_assets` is also high, at **0.904**. 
This means that companies with larger asset bases tend to generate higher revenue.
Again, this reflects the size of a bissiness.
Overall, the strongest correlations are mostly found between absolute financial indicators. 
This is expected because absolute financial values are strongly influenced by company size - which is also can be observed loking at rhe skewness of data.
For this reason, high correlation does not necessarily mean a causal relationship.
To reduce the influence of scale, i will later analyze it after KDE decomposition and building correlation
among separately "big" and "small" companies and with diviasion of it into regions and movement of stock proces during observed period in given data.

Some correlations should be interpreted with extra caution. For example, `employees_year` has a high correlation with `return_1y`, `return_90d`, `net_margin`, and `operating_margin`. 
This relationship is less economically intuitive but it occures because of a high number of missed values in 'employees_year'. So, it is also not that matter.
The correlation between `cumulative_return` and `cagr` is also expected because both variables describe stock price performance over time. CAGR is a normalized annual growth measure, while cumulative return describes total growth over the whole period. Since they are mathematically related, a positive correlation between them is natural.

In conclusion, the correlation matrix shows that many financial variables move together, 
mainly because they are connected through company size and accounting structure. 

""")

    elif selected_dataset == "Annual financials":
        st.caption("""Missed values""")
        st.markdown("""


The dataset containing annual financial metrics of companies has several columns with missing values. 
Before applying any imputation or removing observations, 
it is important to understand the nature of these missing data.

As a first step, we will check which companies contain missing values and whether the same companies 
repeatedly appear across different variables. 
This will allow us to determine whether the problem is company-specific or affects the dataset more broadly.

We will also analyze the distribution of missing values over time.
In financial datasets, missing observations often occur during the earliest years of 
a company's reporting history. In such cases, 
the missing values may simply means that the company had not yet reached a sufficient reporting scale
 or that the corresponding information was not disclosed at that time. If a particular year contains very limited information across most financial indicators, removing such observations may be more appropriate than attempting to impute them.

The variable with the largest number of missing values is `research_dev`, which describes research and development expenditures. To better understand these missing observations, we investigate whether they are associated with specific company segments, industries, or company sizes. Since R&D spending is not equally relevant for all business models, the absence of reported values may reflect differences in company characteristics rather than data quality issues.

For each variable with missing observations, we examine the affected companies individually and evaluate whether the missing values follow a systematic pattern. Only after understanding the source of the missing data do we decide on the appropriate treatment strategy, which may include removing observations, creating indicator variables, deriving values from related financial metrics, or leaving the values as missing when they carry meaningful information.

This approach helps preserve the economic interpretation of the dataset while minimizing the risk of introducing bias through inappropriate data imputation.


""")
        st.caption("""Skewness""")
        st.markdown("""
The results show that almost all financial variables have important skewness. 
The highest values are observed for capex (12.29), operating_income (11.46), 
ebitda (10.92), net_income (10.84), and free_cash_flow (10.26). 
Such extremely high skewness indicates that a small number of observations are much larger 
than the majority of the dataset.

This pattern is expected in a dataset
containing companies of very different sizes.
Large global e-commerce firms generate revenues, profits, assets, and cash flows that are several orders of magnitude greater than those of smaller companies. As a result, most observations are concentrated at relatively low values, while a small number of very large companies create a long right tail in the distribution.

The same effect can be observed for balance-sheet variables such as total_assets (8.27), total_equity (10.22), and total_liabilities (4.71). These indicators are strongly influenced by company scale and therefore naturally exhibit highly asymmetric distributions.

Even research_dev demonstrates considerable skewness (4.40). This suggests that while many companies report relatively modest research and development expenditures, a small number of firms invest substantially more in innovation activities than the rest of the sample.

The only variable with an approximately symmetric distribution is fiscal_year, with a skewness coefficient close to zero (0.04). This is expected because observations are distributed relatively evenly across the analyzed time period.

""")
    elif selected_dataset == "Macro indicators":
        st.caption("""Missed values""")
  

    # ─────────────────────────────
    # DESCRIPTIVE STATISTICS
    # ─────────────────────────────

    st.subheader("2. Descriptive statistics for selected numerical fields")

    key_stats_cols = {
        "Merged companies + metrics": [
            "cagr",
            "cumulative_return",
            "annualized_volatility",
            "max_drawdown"
        ],
        "Annual financials": [
            "total_revenue",
            "net_income",
            "total_assets",
            "free_cash_flow"
        ],
        "Macro indicators": [
            "value"
        ]
    }

    cols = [
        col for col in key_stats_cols[selected_dataset]
        if col in selected_df.columns
    ]

    if len(cols) > 0:
        stats_table = pd.DataFrame({
            "mean": selected_df[cols].mean(),
            "median": selected_df[cols].median(),
            "std": selected_df[cols].std(),
            "p25": selected_df[cols].quantile(0.25),
            "p75": selected_df[cols].quantile(0.75)
        }).round(4)

        st.dataframe(stats_table, use_container_width=True)
    if selected_dataset == "Merged companies + metrics":
        st.markdown("""
### Interpretation of the Descriptive Statistics

The table presents descriptive statistics for four main indicators of post-IPO stock performance: `cagr`, `cumulative_return`, `annualized_volatility`, and `max_drawdown`. The mean, median, standard deviation, 25th percentile, and 75th percentile are shown for each indicator.

#### Compound Annual Growth Rate

The mean CAGR is 3.48% per year. The median is slightly lower and equals 2.76% per year. This means that the typical company in the dataset had a small positive annual stock return.

The mean and the median are relatively close. However, the standard deviation is 0.1877, or 18.77 percentage points. Therefore, a large difference between the annual growth rates of individual companies can still be observed.

The 25th percentile is -0.0736. This means that 25% the companies had a CAGR below approximately -7.36% per year. The 75th percentile is 0.1552, which means that 25% of the companies had a CAGR above approximately 5.52% per year.

The middle 50% of the observations are located between -7.36% and 15.52%. This range includes both negative and positive values. Therefore, long-term stock performance was not positive for all companies.

#### Cumulative Return

The mean cumulative return is approximately 427.27%. However, the median is only  21.21%.

A very large difference between the mean and the median can be seen. The mean is more than twenty times higher than the median. This indicates that the distribution is strongly affected by a small number of companies with very high cumulative returns.

The standard deviation is 1.6046. This is the highest standard deviation among the four indicators. It shows that cumulative stock performance differs greatly across companies.

The 25th percentile is −0.5455. Therefore, at least 25% of the companies had a cumulative return below −54.55%. The 75th percentile is  252.03%.

As a result, the middle 50% of the companies had cumulative returns between **−54.55% and 252.03%**. This is a very wide interval. Some companies lost a large part of their stock value, while other companies produced returns of more than 250%.

The median cumulative return of **21.21%** gives a more realistic description of a typical company than the mean of **427.27%**. The mean was strongly increased by the best-performing companies.

#### Annualized Volatility

The mean annualized volatility is **0.5374**, or approximately **53.74%**. The median is **0.5201**, or **52.01%**.

The mean and median are close to each other. This suggests that the volatility distribution is more balanced than the cumulative return distribution.

The standard deviation is **0.1509**, or approximately 15.09%. Therefore, volatility differs across companies, but the differences are smaller than the differences in cumulative returns.

The 25th percentile is **0.4418**, or **44.18%**, while the 75th percentile is **0.6052**, or **60.52%**. This means that the middle 50% of the companies had annualized volatility between **44.18% and 60.52%**.

These values show that high price variability was common in the dataset. Even the lower quartile of annualized volatility is above 44%. Therefore, post-IPO stock prices were unstable for a large part of the companies.

#### Maximum Drawdown

The mean maximum drawdown is −78.91%. The median is even lower and equals −82.49%.

A maximum drawdown represents the largest decline from a previous price peak. Therefore, a value of −82.49% means that the median company experienced a decline of 82.49% from its highest previous price level.

The standard deviation is 14.78%. The 25th percentile is −0.8752, while the 75th percentile is −0.7164.

This means that the middle 50% of companies had maximum drawdowns between approximately −87.52% and −71.64%. Even the 75th percentile is equal to −71.6%, which shows that very large price declines were common across the dataset.

The median drawdown is more negative than the mean drawdown. This suggests that many companies experienced very deep losses, while a smaller number of companies had less severe drawdowns and increased the mean value.

#### General Conclusion

The descriptive statistics show that post-IPO stock performance differs strongly across e-commerce companies.

The average CAGR is positive and equals 3.48%, but the middle 50% of companies have CAGR values between −7.36% and 15.52%. Therefore, positive long-term growth was not observed for every company.

Cumulative returns show the largest variation. The mean cumulative return is **427.27%**, while the median is only **21.21%**. This large difference shows that the mean was strongly affected by a small number of very successful companies.

Annualized volatility is also high. The median company has volatility of 52.01%, and the middle 50% of observations are located between 44.18% and 60.52%.

Maximum drawdowns are especially large. The median maximum drawdown is −82.49%, and 75% of the companies have a drawdown equal to or more negative than −71.64%.

Overall, the results show that the stocks in the dataset were associated with high risk. Some companies produced very high returns, but large price fluctuations and deep losses were also common.

""")
    elif selected_dataset == "Annual financials":
        st.markdown("""
Interpretation of the Financial Descriptive Statistics
The table presents descriptive statistics for total revenue, net income, total assets, and free cash flow. The values are measured in US dollars. Large differences between the mean and the median can be seen for all four indicators. This shows that the results are strongly affected by a small number of very large observations.
Total revenue
The mean total revenue is approximately 493.46 billion USD, while the median is only 12.45 billion USD. The mean is approximately 3,862.49% higher than the median. This large difference shows that several company-year observations have extremely high revenue values. The standard deviation is approximately 2.33 trillion USD. It is equal to about 471.17% of the mean. Therefore, revenue values are spread over a very wide range. The lower 25% of observations have revenue of 4.35 billion USD or less. The upper 25% have revenue of 64.54 billion USD or more. The threshold for the upper 25% is approximately 1,383.46% higher than the threshold for the lower 25%. The middle 50% of the observations are located between 4.35 billion USD and 64.54 billion USD. This range is much lower than the mean of 493.46 billion USD. Therefore, the mean does not describe a typical observation well. The median gives a more realistic value for the main part of the dataset.
Net income
The mean net income is approximately −792.34 billion USD, while the median is positive and equals approximately 228.35 million USD. The mean and the median have different signs. Therefore, a percentage comparison between them would not give a clear result. The negative mean was probably caused by a small number of extremely large losses. The standard deviation is approximately 7.52 trillion USD. It is equal to about 948.47% of the absolute value of the mean. This shows that net income differs greatly across company-year observations. The lower 25% of observations have net income of −336.83 million USD or less. The upper 25% have net income of 1.98 billion USD or more. The middle 50% of observations are located between a loss of 336.83 million USD and a profit of 1.98 billion USD. Since this interval includes both negative and positive values, both losses and profits are common in the dataset. The positive median means that at least 50% of observations are located at or above approximately 228.35 million USD. However, several very large negative values reduced the mean to −792.34 billion USD.
Total assets
The mean total assets are approximately 2.33 trillion USD, while the median is approximately 19.29 billion USD. The mean is approximately 11,982.84% higher than the median. This is the largest percentage difference between the mean and the median among the four indicators. It shows that a small number of observations have much larger asset values than the rest of the dataset. The standard deviation is approximately 12.75 trillion USD. It is equal to about 547.30% of the mean. Therefore, very large differences in company size can be seen. The lower 25% of observations have total assets of 4.79 billion USD or less. The upper 25% have total assets of 80.67 billion USD or more.
The threshold for the upper 25% is approximately 1,583.58% higher than the threshold for the lower 25%. The middle 50% of observations are located between 4.79 billion USD and 80.67 billion USD. The median of 19.29 billion USD is much lower than the mean of 2.33 trillion USD. Therefore, the mean is strongly affected by the largest companies. The median gives a better description of a typical company-year observation.
Free cash flow
The mean free cash flow is approximately −164.31 billion USD, while the median is positive and equals approximately 449.90 million USD. As with net income, the mean and the median have different signs. A percentage comparison between them would be misleading. The negative mean shows that several very large negative cash flow values had a strong effect on the result. The standard deviation is approximately 1.49 trillion USD. It is equal to about 908.77% of the absolute value of the mean. This shows that free cash flow values are highly different across the dataset.
The lower 25% of observations have free cash flow of −138.25 million USD or less. The upper 25% have free cash flow of 2.44 billion USD or more. The middle 50% of observations are located between negative 138.25 million USD and positive 2.44 billion USD. This interval crosses zero. Therefore, both negative and positive free cash flow values are present in the central part of the dataset. The positive median shows that at least 50% of observations have free cash flow at or above approximately 449.90 million USD. However, the mean is negative because it was reduced by several very large negative observations.
General conclusion
Large differences between the mean and the median are observed for all four financial indicators. For total revenue, the mean is approximately 3,862.49% higher than the median. For total assets, the mean is approximately 11,982.84% higher than the median. These results show that a small number of very large companies strongly affect the average values.
Net income and free cash flow have positive medians but negative means. This means that the typical observation is positive, while several very large losses reduce the average below zero.
The standard deviations are also very high. They represent approximately 471.17% of the mean revenue, 547.30% of the mean total assets, 948.47% of the absolute mean net income, and 908.77% of the absolute mean free cash flow.

""")
    elif selected_dataset == "Macro indicators":
        st.markdown("""
The column value contains the largest number of IQR-based outliers. In this dataset, outliers are not automatically treated as errors: for financial and market data, extreme values often represent real differences in company size, profitability or market performance. Therefore, the plot is used mainly for diagnosis, while final removal decisions are discussed in the data-cleaning section.
The IQR method identified 270 outliers in the value column. These observations represent 19.07% of the macroeconomic dataset. The calculated lower boundary is approximately −53.71 million, while the upper boundary is approximately 89.52 million.
Values outside these boundaries were classified as outliers. However, these results should be interpreted very carefully. A value above 89.52 million may be normal for population, total GDP, or another large-scale indicator. At the same time, a value of this size would be impossible for a percentage indicator.
Therefore, the 270 identified outliers do not necessarily represent incorrect or unusual observations. Many of them are classified as outliers only because indicators with different scales were combined in one column. The year column contains no outliers. Its calculated boundaries are 2001 and 2033. All observations are located inside this interval.
However, the year variable is used as a time identifier. It is not treated as an economic measure. Therefore, its mean, quartiles, and IQR boundaries are not important for the economic interpretation of the dataset.
Box Plots Before and After Filtering
The box plot before filtering shows a highly compressed distribution. The vertical axis uses a scale of (10^13). Most observations are located close to zero on this scale. Many large observations are shown above the main box. The highest values are close to 29 trillion. These values strongly affect the plot. Because of them, percentage indicators and other smaller values cannot be clearly seen. After IQR filtering, the scale is reduced from (10^13) to (10^7). However, the filtered distribution is still not easy to interpret. Several groups of large observations remain. Some values are located near 35–50 million, others near 63–68 million, and another group is located near 80–84 million.
At the same time, most smaller values are still compressed near zero.
The second plot also recalculates the quartiles after the original outliers are removed. Therefore, some observations can still be displayed as separate points.
The filtering improves the scale of the graph, but it does not solve the main problem. Different indicators are still stored together in the same numerical column.
Why Data Transformation Is Required
The original macroeconomic dataset must be transformed before meaningful descriptive statistics can be calculated. The indicator_name column contains the name of each macroeconomic factor. The corresponding numerical result is stored in the value column. For example, one row may contain gdp_total_usd in indicator_name and its monetary value in value. Another row may contain internet_users_pct and a percentage value in the same value column.
These two values should not be analysed as observations of the same variable. The dataset should first be transformed from long format into wide format. After the transformation, each macroeconomic indicator should have its own column.
For example, separate columns can be created for:
gdp_total_usd; gdp_per_capita_usd; internet_users_pct; population_total; urban_population_pct; account_ownership_pct_adult.
        
""")


    st.subheader("3. Outlier scan using IQR rule")

    st.markdown("""
    Outliers are detected using the standard IQR rule:

    lower bound = Q1 − 1.5 × IQR  
    upper bound = Q3 + 1.5 × IQR

    This scan is applied to every numerical column in the selected raw dataset.
    """)

    outliers_result = outlier_scan(selected_df)

    st.dataframe(
        outliers_result.head(10),
        use_container_width=True
    )

    if not outliers_result.empty:
        worst_col = outliers_result.iloc[0]["column"]

        st.write(f"Column with the largest number of outliers: **{worst_col}**")

        outliers, lower, upper = find_outliers(selected_df[worst_col])

        before = selected_df[worst_col].dropna()
        after = before[(before >= lower) & (before <= upper)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.boxplot(y=before, ax=axes[0])
        axes[0].set_title(f"Before IQR filtering\n{worst_col}")
        axes[0].set_ylabel(worst_col)

        sns.boxplot(y=after, ax=axes[1])
        axes[1].set_title(f"After IQR filtering\n{worst_col}")
        axes[1].set_ylabel(worst_col)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
        The column `{worst_col}` contains the largest number of IQR-based outliers.
        In this dataset, outliers are not automatically treated as errors: for financial and market data,
        extreme values often represent real differences in company size, profitability or market performance.
        Therefore, the plot is used mainly for diagnosis, while final removal decisions are discussed
        in the data-cleaning section.
        """)
        if selected_dataset == "Merged companies + metrics":
            st.markdown(r"""
    ### IQR Filtering of `total_assets`

    The figure compares the distribution of `total_assets` before and after
    IQR filtering. The purpose of this step was to identify extremely large
    values and to check hw strongly they affected the general distribution

    #### Distribution before filtering

The box plot on the left shows all 43 observations from the original dataset. The values of total_assets are measured in US dollars. Scientific notation is used on the vertical axis. The value 1e13 means that the numbers shown on the axis are multiplied by (10^{13}).

A very large difference between companies can be seen in the original data. Most observations are located close to the bottom of the plot. At the same time, several companies have much larger asset values.

The highest value is approximately 45.76 trillion USD, while the second-highest value is approximately 28.80 trillion USD. Several other observations are also much higher than the main group. Because of these extreme values, the box itself is compressed near zero. As a result, the distribution of most companies cannot be clearly seen in the first plot.

Before filtering, the main descriptive statistics were:

the minimum value was approximately 0.43 billion USD;
the first quartile was approximately 6.67 billion USD;
the median was approximately 19.72 billion USD;
the third quartile was approximately 87.82 billion USD;
the maximum value was approximately 45.76 trillion USD.

The mean value was approximately 1.87 trillion USD. This value was much higher than the median. The difference between the mean and the median was caused by the small number of extremely large observations. Therefore, the mean did not describe a typical company in this dataset very well.

Calculation of the IQR boundaries

The IQR method was used to identify outliers. First, the interquartile range was calculated:

IQR = Q_3 - Q_1

For total_assets, the first quartile was approximately 6.67 billion USD, and the third quartile was approximately 87.82 billion USD. Therefore, the IQR was approximately 81.15 billion USD.

The lower and upper boundaries were calculated with the following formulas:


Lower_bound = Q_1 - 1.5 \times IQR

Upper_bound = Q_3 + 1.5 \times IQR


The resulting lower boundary was approximately −115.07 billion USD, while the upper boundary was approximately 209.55 billion USD.

Since company assets cannot normally be negative, no observations were identified below the lower boundary. However, 9 observations were located above the upper boundary. These observations represented approximately 20.93% of all companies in the dataset.

These values were classified as statistical outliers and were excluded from the filtered distribution. However, they should not automatically be treated as data errors. They may represent very large companies that are different from the rest of the sample.

Distribution after filtering

The box plot on the right shows the distribution after the 9 observations above the original IQR boundary were removed. A total of 34 observations remained in the dataset.

After filtering, the scale of the vertical axis was reduced from (10^{13}) to (10^{11}). Therefore, the central part of the distribution can be seen much more clearly.

The descriptive statistics after filtering were:

the minimum value was approximately 0.43 billion USD;
the first quartile was approximately 3.50 billion USD;
the median was approximately 16.40 billion USD;
the third quartile was approximately 39.73 billion USD;
the maximum value was approximately 120.63 billion USD.

The mean value after filtering was approximately 27.34 billion USD. It became much closer to the median than before filtering. This shows that the extremely large companies had a strong effect on the original mean.

The box in the second plot covers the middle 50% of the filtered observations. Most companies have total assets between approximately 3.50 billion USD and 39.73 billion USD. The median is located at approximately 16.40 billion USD. The upper whisker extends to approximately 88.58 billion USD.

One observation of approximately 120.63 billion USD is still shown as a separate point in the second box plot. This does not mean that the filtering was performed incorrectly. The original IQR boundaries were calculated before the extreme observations were removed. After filtering, the quartiles were calculated again by the box plot. A new and narrower distribution was produced. Therefore, the value of 120.63 billion USD is treated as an outlier relative to the new filtered distribution, even though it was inside the original upper boundary of 209.55 billion USD.
    """)
        elif selected_dataset == "Annual financials":
            st.markdown("""
      The results show that outliers are present in all selected financial indicators. Their share varies from 14.53% to 24.42%. This means that large differences between company-year observations are common in the dataset.

These outliers should not automatically be treated as incorrect values. The dataset contains companies of different sizes, countries, and business models. Therefore, large differences in revenue, assets, income, and cash flow can be expected.

General Outlier Results

The highest outlier share was found in free_cash_flow. A total of 42 observations were classified as outliers. This represents 24.42% of all available observations for this variable.

The IQR boundaries for free_cash_flow were approximately −4.01 billion USD and 6.31 billion USD. Therefore, values below −4.01 billion USD and above 6.31 billion USD were treated as outliers.

The second-highest outlier share was found in net_income. It contained 41 outliers, or 23.84% of the observations. The lower boundary was approximately −381.28 million USD, while the upper boundary was approximately 5.46 billion USD.

For operating_income, 39 observations were classified as outliers. This represents 22.67% of the dataset. The lower boundary was approximately −399.62 million USD, and the upper boundary was approximately 6.12 billion USD.

The capex variable contained 34 outliers, or 19.77% of the observations. Its lower boundary was approximately −3.01 billion USD, while its upper boundary was approximately 1.66 billion USD.

For total_assets, 33 observations were identified as outliers. This represents 19.19% of the observations. The calculated boundaries were approximately −109.03 billion USD and 194.49 billion USD.

The negative lower boundary for total_assets was produced by the IQR formula. It does not mean that negative assets were expected. In this case, only observations above the upper boundary were relevant as possible outliers.

Both cash_and_equivalents and total_equity contained 32 outliers. For each variable, the outlier share was 18.60%.

The IQR boundaries for cash_and_equivalents were approximately −8.00 billion USD and 15.04 billion USD. For total_equity, the boundaries were approximately −79.22 billion USD and 136.73 billion USD.

For operating_cash_flow, 28 observations were classified as outliers. This represents 16.28% of the observations. The boundaries were approximately −701.66 million USD and 11.86 billion USD.

The same number of outliers was found in ebitda. It also contained 28 outliers, or 16.28% of the observations. The lower boundary was approximately −7.33 billion USD, while the upper boundary was approximately 12.38 billion USD.

The lowest outlier share among the indicators shown in the table was found in total_revenue. It contained 25 outliers, which represents 14.53% of the observations.

The lower IQR boundary for total_revenue was approximately −85.93 billion USD, and the upper boundary was approximately 154.81 billion USD. Since revenue is normally non-negative, the negative lower boundary was only a mathematical result of the IQR calculation.

The difference between the highest and the lowest outlier shares is 9.89 percentage points. Therefore, free_cash_flow contains noticeably more extreme observations than total_revenue.

Free Cash Flow Before IQR Filtering

The left box plot shows the original distribution of free_cash_flow.

The vertical axis is shown with a scale of (10^{13}). This means that the values displayed on the axis are multiplied by (10^{13}).

Several extremely large negative observations can be seen. The lowest value is located near −1.76 × (10^{13}), or approximately −17.6 trillion USD. Other very large negative observations are located near −7.1 trillion USD and −4.6 trillion USD.

There are also several large positive and negative values closer to zero. However, because of the extremely large negative observations, the main box is strongly compressed near zero.

As a result, the central part of the distribution cannot be clearly examined in the first plot. The median, quartiles, and main range of the observations are almost invisible.

This shows that a small number of extreme observations have a very strong effect on the scale of the graph.

The original descriptive statistics also support this result. The mean free cash flow was approximately −164.31 billion USD, while the median was positive and equal to approximately 449.90 million USD.

The mean and median have different signs. This happened because a small number of very large negative values reduced the mean far below zero.

The standard deviation was approximately 1.49 trillion USD. This value was more than nine times larger than the absolute value of the mean. Therefore, the original distribution had a very high level of variation.

The 25th percentile was approximately −138.25 million USD, while the 75th percentile was approximately 2.44 billion USD. This means that the middle 50% of observations were located between these two values.

This central range is much smaller than the extreme values shown in the first plot. Therefore, most company-year observations were located much closer to zero than the largest outliers.

Free Cash Flow After IQR Filtering

The right box plot shows the distribution after the original IQR outliers were removed.

A total of 42 observations were excluded. Therefore, 75.58% of the original observations remained in the filtered dataset.

After filtering, the scale of the vertical axis changed from (10^{13}) to (10^9). This is a major reduction in scale.

The central part of the distribution can now be seen much more clearly. The box is no longer compressed near zero.

Most of the retained observations are concentrated around values between slightly below zero and approximately 1.5 billion USD. The median is located above zero, which is consistent with the positive original median of approximately 449.90 million USD.

The lower whisker extends to approximately −2.1 billion USD, while the upper whisker extends to approximately 3.4 billion USD.

Several separate points are still shown below and above the whiskers. Negative points are located approximately between −4.0 billion USD and −2.8 billion USD. Positive points are located approximately between 4.2 billion USD and 6.2 billion USD.

These remaining points do not mean that the original filtering was performed incorrectly.

The original IQR boundaries were calculated using the full dataset. After the most extreme observations were removed, the box plot calculated new quartiles for the filtered data.

The new distribution became narrower. Therefore, some values that were inside the original boundaries can still be shown as outliers relative to the new filtered distribution.

For example, the original upper boundary was approximately 6.31 billion USD. A value close to 6.2 billion USD was not removed because it remained below this original boundary. However, it can still be displayed as an outlier in the second box plot because the quartiles were recalculated after filtering.

Interpretation of the Free Cash Flow Distribution

The comparison between the two plots shows that the original free_cash_flow distribution was strongly affected by a small number of extreme observations.

Before filtering, 24.42% of the values were located outside the IQR boundaries. This was the highest outlier share among all financial indicators included in the analysis.

The largest problem was caused by extremely negative values. These observations changed the scale of the graph and made the main distribution difficult to see.

After the original IQR outliers were removed, the central distribution became much clearer. Most retained observations were located within a range of several billion USD around zero.

The positive median shows that the typical company-year observation had positive free cash flow. However, the negative mean shows that some companies experienced extremely large cash outflows.

Therefore, the median gives a better description of a typical observation than the mean.

The filtered plot is more useful for examining the main part of the distribution. However, the removed observations should not automatically be deleted from every part of the analysis.

Large negative free cash flow can represent real business conditions. It may appear during periods of major investment, rapid expansion, acquisitions, or financial difficulties.

For this reason, the filtered data can be used for visual analysis and for methods that are strongly affected by extreme values. The full data should still be preserved when the complete financial history of the companies is analysed.      
The outlier share is above 14% for every indicator included in the table. For six of the ten indicators, the outlier share is higher than 18%.

The highest shares were found in free_cash_flow, net_income, and operating_income. Their outlier shares were 24.42%, 23.84%, and 22.67%, respectively.

These indicators can contain both positive and negative values. They can also change strongly from one year to another. Therefore, a high number of outliers can be expected.

The lowest outlier share was found in total_revenue, but even this variable contained 25 outliers, or 14.53% of the observations.

The results show that the annual financial dataset is strongly unequal. A small number of very large companies and unusual company-year observations have a strong effect on means, standard deviations, and visual scales.
                        """)
        elif selected_dataset == "Macro indicators":
            st.markdown("""
The IQR method identified 270 outliers in the value column. These observations represent 19.07% of the macroeconomic dataset.

The calculated lower boundary is approximately −53.71 million, while the upper boundary is approximately 89.52 million.

Values outside these boundaries were classified as outliers. However, these results should be interpreted very carefully.

A value above 89.52 million may be normal for population, total GDP, or another large-scale indicator. At the same time, a value of this size would be impossible for a percentage indicator.

Therefore, the 270 identified outliers do not necessarily represent incorrect or unusual observations. Many of them are classified as outliers only because indicators with different scales were combined in one column.

The year column contains no outliers. Its calculated boundaries are 2001 and 2033. All observations are located inside this interval.

However, the year variable is used as a time identifier. It is not treated as an economic measure. Therefore, its mean, quartiles, and IQR boundaries are not important for the economic interpretation of the dataset.

Box Plots Before and After Filtering

The box plot before filtering shows a highly compressed distribution. The vertical axis uses a scale of (10^13).

Most observations are located close to zero on this scale. Many large observations are shown above the main box. The highest values are close to 29 trillion.

These values strongly affect the plot. Because of them, percentage indicators and other smaller values cannot be clearly seen.

After IQR filtering, the scale is reduced from (10^13) to (10^7). However, the filtered distribution is still not easy to interpret.

Several groups of large observations remain. Some values are located near 35–50 million, others near 63–68 million, and another group is located near 80–84 million.

At the same time, most smaller values are still compressed near zero.

The second plot also recalculates the quartiles after the original outliers are removed. Therefore, some observations can still be displayed as separate points.

The filtering improves the scale of the graph, but it does not solve the main problem. Different indicators are still stored together in the same numerical column.

Why Data Transformation Is Required

The original macroeconomic dataset must be transformed before meaningful descriptive statistics can be calculated.

The indicator_name column contains the name of each macroeconomic factor. The corresponding numerical result is stored in the value column.

For example, one row may contain gdp_total_usd in indicator_name and its monetary value in value. Another row may contain internet_users_pct and a percentage value in the same value column.

These two values should not be analysed as observations of the same variable.

The dataset should first be transformed from long format into wide format. After the transformation, each macroeconomic indicator should have its own column.

For example, separate columns can be created for:

gdp_total_usd;
gdp_per_capita_usd;
internet_users_pct;
population_total;
urban_population_pct;
account_ownership_pct_adult.

After this transformation, the descriptive statistics should be calculated separately for every indicator.

A mean GDP value can then be compared with other GDP values. A mean internet usage percentage can be compared with other percentage values. Outliers can also be identified separately for each indicator.
""")
    

    # ─────────────────────────────
    # SKEWNESS ANALYSIS
    # ─────────────────────────────

    st.subheader("4. Distribution analysis: skewness")

    st.markdown("""
        Skewness measures whether a distribution is symmetric or shifted to one side.
        Columns with `|skewness| > 1` are treated as **highly skewed**.
        """)

    macro_raw_wide = (
        df_macro
        .pivot_table(
            index=["country_code", "year"],
            columns="indicator_name",
            values="value")
        .reset_index())

    skew_datasets = {
        "Annual financials": df_fin_raw,
        "Macro indicators": macro_raw_wide}

    selected_skew_dataset = st.selectbox(
        "Choose dataset for skewness analysis",
        list(skew_datasets.keys()))

    skew_df = skew_datasets[selected_skew_dataset]

    numeric_cols = skew_df.select_dtypes(include=np.number).columns

    skew_table = (
        skew_df[numeric_cols]
        .skew()
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .reset_index())

    skew_table.columns = ["column", "skewness"]
    skew_table["highly_skewed"] = skew_table["skewness"].abs() > 1

    st.dataframe(skew_table, use_container_width=True)

    top_skew = skew_table.head(10).copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [
        "tomato" if abs(v) > 1 else "steelblue"
        for v in top_skew["skewness"]
    ]

    bars = ax.bar(
        top_skew["column"],
        top_skew["skewness"],
        color=colors)

    ax.axhline(1, color="red", linestyle="--", alpha=0.6)
    ax.axhline(-1, color="red", linestyle="--", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    ax.set_title(f"Top 10 most skewed columns: {selected_skew_dataset}")
    ax.set_ylabel("Skewness")
    ax.set_xlabel("Numeric columns")

    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    highly_skewed_count = skew_table["highly_skewed"].sum()
    most_skewed_col = skew_table.iloc[0]["column"]
    most_skewed_value = skew_table.iloc[0]["skewness"]

    st.markdown(f"""The most skewed column in **{selected_skew_dataset}** is `{most_skewed_col}` with skewness equal to **{most_skewed_value:.2f}**.
                Overall, **{highly_skewed_count}** numeric columns have `|skewness| > 1`.
                This means that their distributions are substantially asymmetric.""")

    if selected_skew_dataset == "Annual financials":
        st.markdown("""
                    For annual financial statements, high skewness is expected because companies differ greatly in scale.
                    Variables such as revenue, assets, free cash flow or capex may vary by several orders of magnitude.
                    Also, some financial indicators can be negative, so log transformation is not always appropriate.
                    For example, capex and free cash flow may contain negative values, which makes a simple logarithmic
                    transformation unsuitable.""")

    else:
        st.markdown("""For macroeconomic indicators, high skewness usually reflects structural differences between countries.
                    For example, GDP per capita or FDI inflows may differ strongly across economies. Unlike firm-level
                    financial data, macro indicators are usually cleaner, but their distributions may still be asymmetric
                    because countries are heterogeneous in income level, population structure and economic development.""")

# ══════════════════════════════════════════
# 4. DATA CLEANUP
# ══════════════════════════════════════════

elif page == "Data Cleanup":
    st.title("Data Cleanup")

    st.markdown("""
    ## 4. Data Cleanup

    Data cleanup was performed separately for three main parts of the project:

    - company and market performance data,
    - annual financial statements,
    - macroeconomic indicators.

    The goal was not only to remove missing values, but also to understand why they appeared
    and whether they could be correctly reconstructed from other available columns.
    """)

    # ─────────────────────────────────────
    # Small display helpers used only on this page
    # ─────────────────────────────────────

    def show_value_comparison(title, calculated, dataset_value):
        """Display one formula-verification result as a compact card."""
        with st.container(border=True):
            st.markdown(f"#### {title}")
            col_calc, col_data, col_diff = st.columns(3)

            col_calc.metric(
                "Calculated value",
                f"{calculated:.4f}"
            )
            col_data.metric(
                "Dataset value",
                f"{dataset_value:.4f}"
            )
            col_diff.metric(
                "Absolute difference",
                f"{abs(calculated - dataset_value):.6f}"
            )


    def show_missing_values(missing_series, total_rows, empty_message):
        """Display missing-value counts and shares in a readable table."""
        if missing_series.empty:
            st.success(empty_message)
            return

        missing_table = (
            missing_series
            .rename("Missing values")
            .reset_index()
            .rename(columns={"index": "Column"})
        )
        missing_table["Share"] = (
            missing_table["Missing values"] / total_rows * 100
        )
        missing_table = missing_table.sort_values(
            "Missing values",
            ascending=False
        )

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Column": st.column_config.TextColumn("Column"),
                "Missing values": st.column_config.NumberColumn(
                    "Missing values",
                    format="%d"
                ),
                "Share": st.column_config.ProgressColumn(
                    "Share of rows",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0
                )
            }
        )

    tab_company, tab_financial, tab_macro = st.tabs([
        "Company & Market Data",
        "Financial Metrics",
        "Macroeconomic Indicators"
    ])

    # ═════════════════════════════════════
    # 4.1 Company + Market Metrics
    # ═════════════════════════════════════

    with tab_company:
        st.header("4.1 Company and Market Performance Data")

        st.markdown("""
        The first cleaned dataset was obtained by merging company descriptions with company-level
        market performance metrics.

        Several columns were removed because they had either high missingness or low analytical
        value for the following quantitative analysis. These included textual metadata, Wikipedia
        links, company descriptions, founders, headquarters, and employee-related columns.

        After that, missing values in stock performance metrics were investigated. The missing values
        appeared in the same set of companies for the following variables:

        - `last_close`
        - `cumulative_return`
        - `cagr`
        - `return_1y`
        - `return_90d`

        Instead of dropping these companies, the formulas for these metrics were reconstructed and
        checked on companies where the values were already available.
        """)

        with st.container(border=True):
            st.markdown("### Formulas used")
            formula_col1, formula_col2 = st.columns(2)

            with formula_col1:
                st.markdown("""
                **Cumulative return**

                $$
                cumulative\\ return =
                \\frac{last\\ close}{first\\ close} - 1
                $$
                """)

                st.markdown("""
                **CAGR**

                $$
                CAGR =
                \\left(
                \\frac{last\\ close}{first\\ close}
                \\right)^{\\frac{1}{years}} - 1
                $$
                """)

            with formula_col2:
                st.markdown("""
                **1-year return and 90-day return**

                $$
                return =
                \\frac{adjusted\\ close_t}
                {adjusted\\ close_{t-k}} - 1
                $$
                """)

        # ─────────────────────────────────
        # Formula verification on one sample
        # ─────────────────────────────────

        st.subheader("Formula verification")

        sample = df[df["cagr"].notna()].iloc[0].copy()
        company_id = sample["company_id"]

        sample_fields = [
            column for column in [
                "company_id",
                "name",
                "ticker_x",
                "ticker",
                "history_start",
                "history_end",
                "first_close",
                "last_close",
                "cumulative_return",
                "cagr",
                "return_1y",
                "return_90d"
            ]
            if column in sample.index
        ]

        with st.expander(
            f"Validation sample: {company_id}",
            expanded=False
        ):
            st.caption(
                "A company with non-missing market metrics was selected "
                "to verify the reconstructed formulas."
            )
            st.dataframe(
                sample[sample_fields]
                .rename("Value")
                .to_frame(),
                use_container_width=True
            )

        # Cumulative return
        calc_cumulative = (
            sample["last_close"] / sample["first_close"]
        ) - 1
        actual_cumulative = sample["cumulative_return"]

        show_value_comparison(
            "Cumulative return",
            calc_cumulative,
            actual_cumulative
        )

        # CAGR
        history_start = pd.to_datetime(sample["history_start"])
        history_end = pd.to_datetime(sample["history_end"])
        years = (history_end - history_start).days / 365.25

        calc_cagr = (
            (sample["last_close"] / sample["first_close"])
            ** (1 / years)
            - 1
        )
        actual_cagr = sample["cagr"]

        show_value_comparison(
            "CAGR",
            calc_cagr,
            actual_cagr
        )

        # Daily stock history for the selected company
        tmp = (
            df_stick[df_stick["company_id"] == company_id]
            .sort_values("date")
            .copy()
        )
        tmp["date"] = pd.to_datetime(tmp["date"])

        last_date = tmp.iloc[-1]["date"]
        last_price = tmp.iloc[-1]["close"]

        # The original 128-day calculation is intentionally preserved.
        target_90d = last_date - pd.DateOffset(days=128)
        price_90d = tmp.loc[
            tmp["date"] <= target_90d,
            "close"
        ].iloc[-1]

        calc_90d = (last_price / price_90d) - 1
        actual_90d = sample["return_90d"]

        show_value_comparison(
            "90-day return",
            calc_90d,
            actual_90d
        )

        with st.expander(
            "Why was a 128-day window used to reconstruct the 90-day return?",
            expanded=False
        ):
            st.markdown("""
            When validating the formula for `return_90d`, it became clear that the stock price dataset
            does not contain observations for every calendar day. Missing dates occur because of weekends,
            public holidays, exchange-specific trading calendars, and differences between countries.

            To identify how the original `return_90d` metric was calculated, the formula was reversed:

            $$
            Price_{t-90} = \\frac{Price_t}{1 + Return_{90D}}
            $$

            Using this reconstructed target price, the closest historical observation was found in the
            daily price series. The difference between this date and the latest available trading date
            was then measured.

            The analysis showed that the reported 90-day returns correspond most closely to a lookback
            window of approximately **128 calendar days**, which accounts for non-trading days in the
            underlying stock price data. Therefore, a 128-day offset was used when reconstructing
            missing `return_90d` values.
            """)

            target_price = last_price / (1 + sample["return_90d"])
            closest = tmp.iloc[
                (tmp["close"] - target_price).abs().argmin()
            ]
            days_diff = (last_date - closest["date"]).days

            date_col1, date_col2, date_col3 = st.columns(3)
            date_col1.metric(
                "Latest trading date",
                str(last_date.date())
            )
            date_col2.metric(
                "Matched historical date",
                str(closest["date"].date())
            )
            date_col3.metric(
                "Calendar-day difference",
                days_diff
            )

        # 1-year return
        target_1y = last_date - pd.DateOffset(years=1)
        price_1y = tmp.loc[
            tmp["date"] <= target_1y,
            "close"
        ].iloc[-1]

        calc_1y = (last_price / price_1y) - 1
        actual_1y = sample["return_1y"]

        show_value_comparison(
            "1-year return",
            calc_1y,
            actual_1y
        )

        # ─────────────────────────────────
        # Full-dataset verification
        # ─────────────────────────────────

        st.subheader(
            "Final formula check on all companies with available values"
        )

        validation_results = []

        # Cumulative return check
        mask = df["cumulative_return"].notna()
        calc = (
            df.loc[mask, "last_close"] /
            df.loc[mask, "first_close"]
        ) - 1

        comparison = pd.DataFrame({
            "actual": df.loc[mask, "cumulative_return"],
            "calculated": calc
        })
        comparison["diff"] = (
            comparison["actual"] - comparison["calculated"]
        ).abs().round(10)

        validation_results.append({
            "Metric": "Cumulative return",
            "Max absolute difference": comparison["diff"].max(),
            "Mean absolute difference": comparison["diff"].mean()
        })

        # CAGR check
        mask = df["cagr"].notna()
        history_start = pd.to_datetime(
            df.loc[mask, "history_start"]
        )
        history_end = pd.to_datetime(
            df.loc[mask, "history_end"]
        )
        years = (history_end - history_start).dt.days / 365.25

        calc = (
            (
                df.loc[mask, "last_close"] /
                df.loc[mask, "first_close"]
            ) ** (1 / years)
        ) - 1

        comparison = pd.DataFrame({
            "actual": df.loc[mask, "cagr"],
            "calculated": calc
        })
        comparison["diff"] = (
            comparison["actual"] - comparison["calculated"]
        ).abs().round(10)

        validation_results.append({
            "Metric": "CAGR",
            "Max absolute difference": comparison["diff"].max(),
            "Mean absolute difference": comparison["diff"].mean()
        })

        # Return 1Y check
        mask = df["return_1y"].notna()
        calc_returns = []
        i = 0

        for idx in df.loc[mask].index:
            company_id = df.loc[idx, "company_id"]
            tmp = (
                df_stick[df_stick["company_id"] == company_id]
                .sort_values("date")
                .copy()
            )
            tmp["date"] = pd.to_datetime(tmp["date"])

            try:
                last_date = tmp.iloc[-1]["date"]
            except Exception:
                print(tmp)
                print("iteration:", i)
                print("idx:", idx)
                print("company_id:", company_id)
                print(df.loc[idx])
                break

            last_price = tmp.iloc[-1]["adj_close"]
            i += 1

            target_date = last_date - pd.DateOffset(years=1)
            price_1y = tmp.loc[
                tmp["date"] <= target_date,
                "adj_close"
            ].iloc[-1]

            calc_returns.append((last_price / price_1y) - 1)

        comparison = pd.DataFrame({
            "actual": df.loc[mask, "return_1y"].values,
            "calculated": calc_returns
        })
        comparison["diff"] = (
            comparison["actual"] - comparison["calculated"]
        ).abs().round(10)

        validation_results.append({
            "Metric": "Return 1Y",
            "Max absolute difference": comparison["diff"].max(),
            "Mean absolute difference": comparison["diff"].mean()
        })

        # Return 90D check
        mask = df["return_90d"].notna()
        calc_returns = []

        for idx in df.loc[mask].index:
            company_id = df.loc[idx, "company_id"]

            tmp = (
                df_stick[df_stick["company_id"] == company_id]
                .sort_values("date")
                .reset_index(drop=True)
            )
            tmp["date"] = pd.to_datetime(tmp["date"])

            last_date = tmp.iloc[-1]["date"]
            last_price = tmp.iloc[-1]["adj_close"]

            # The original 128-day calculation is intentionally preserved.
            target_90d = last_date - pd.DateOffset(days=128)
            price_90d = tmp.loc[
                tmp["date"] <= target_90d,
                "adj_close"
            ].iloc[-1]

            calc_returns.append((last_price / price_90d) - 1)

        comparison = pd.DataFrame({
            "actual": df.loc[mask, "return_90d"].values,
            "calculated": calc_returns
        })
        comparison["diff"] = (
            comparison["actual"] - comparison["calculated"]
        ).abs().round(10)

        validation_results.append({
            "Metric": "Return 90D",
            "Max absolute difference": comparison["diff"].max(),
            "Mean absolute difference": comparison["diff"].mean()
        })

        validation_table = pd.DataFrame(validation_results)

        st.dataframe(
            validation_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric"),
                "Max absolute difference": st.column_config.NumberColumn(
                    "Max absolute difference",
                    format="%.6f"
                ),
                "Mean absolute difference": st.column_config.NumberColumn(
                    "Mean absolute difference",
                    format="%.6f"
                )
            }
        )

        st.markdown("""
        After formula verification, the missing values were restored using daily stock price data.
        For return-based indicators, adjusted closing prices were used because they account for
        splits and other corporate actions.

        The `founded_date` column was also cleaned. Four missing foundation dates were checked
        manually using open sources. In addition, malformed dates containing `00` as month or day
        were converted into valid dates by replacing the unknown part with `01`.
        """)

        st.subheader("Missing values before and after cleaning")

        missing_before = df.isna().sum()
        missing_before = missing_before[missing_before > 0]

        missing_after = df_filled.isna().sum()
        missing_after = missing_after[missing_after > 0]

        missing_col1, missing_col2 = st.columns(2)

        with missing_col1:
            st.markdown("#### Before cleaning")
            show_missing_values(
                missing_before,
                len(df),
                "No missing values were present before cleaning."
            )

        with missing_col2:
            st.markdown("#### After cleaning")
            show_missing_values(
                missing_after,
                len(df_filled),
                "No missing values remain in the cleaned company dataset."
            )

    # ═════════════════════════════════════
    # 4.2 Financial Metrics
    # ═════════════════════════════════════

    with tab_financial:
        st.header("4.2 Financial Metrics Data")

        st.markdown("""
        The annual financial dataset was cleaned separately. First, missing values were checked
        across the main financial indicators:

        - `net_income`
        - `operating_income`
        - `total_revenue`
        - `ebitda`
        - `total_assets`
        - `total_liabilities`
        - `cash_and_equivalents`
        - `total_equity`
        - `capex`
        - `free_cash_flow`

        The analysis showed that rows with missing values in these key columns belonged to the same
        small group of company-year observations. These rows corresponded to the first available
        fiscal year for those companies and contained almost no useful financial information.
        Therefore, they were removed from further financial analysis.
        """)

        critical_cols = [
            "net_income",
            "operating_income",
            "total_revenue",
            "ebitda",
            "total_assets",
            "total_liabilities",
            "cash_and_equivalents",
            "total_equity",
            "capex",
            "free_cash_flow"
        ]

        removed_financial_rows = df_fin_raw[
            df_fin_raw[critical_cols].isna().all(axis=1)
        ][["company_id", "ticker", "fiscal_year"]]

        with st.container(border=True):
            summary_col1, summary_col2 = st.columns(2)
            summary_col1.metric(
                "Rows removed",
                len(removed_financial_rows)
            )
            summary_col2.metric(
                "Companies affected",
                removed_financial_rows["company_id"].nunique()
            )

        with st.expander(
            "Removed financial observations",
            expanded=True
        ):
            st.markdown("""
            These observations were removed because all key financial metrics were missing.
            """)
            st.dataframe(
                removed_financial_rows,
                use_container_width=True,
                hide_index=True
            )

        companies_with_removed_rows = [
            "cartrade",
            "indiamart",
            "ocado"
        ]

        with st.expander(
            "Financial history of affected companies",
            expanded=False
        ):
            st.markdown("""
            To verify whether the removed observations represented missing reporting periods
            or isolated data quality issues, the full financial history of the affected companies
            was inspected.

            As shown below, the removed rows correspond to the first fiscal year available for
            each company. Subsequent years contain complete financial statements, suggesting that
            these observations represent the beginning of the reporting history rather than
            randomly missing data.
            """)

            st.dataframe(
                df_fin_raw[
                    df_fin_raw["company_id"].isin(
                        companies_with_removed_rows
                    )
                ].sort_values(["company_id", "fiscal_year"]),
                use_container_width=True,
                hide_index=True
            )

        st.divider()
        st.subheader("Gross profit")

        st.markdown("""
        Missing values in `gross_profit` were treated differently. They were mainly observed for
        marketplace or financial-service companies, where gross profit is not always reported in
        the same way as for traditional retail businesses.
        """)

        missing_gross_profit = (
            df_fin[df_fin["gross_profit"].isna()][
                ["company_id", "ticker", "fiscal_year"]
            ]
            .merge(
                df_comp[["company_id", "industry"]],
                on="company_id",
                how="left"
            )
        )

        with st.expander(
            "Observations with missing gross profit",
            expanded=False
        ):
            st.dataframe(
                missing_gross_profit,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("""
        Therefore, instead of reconstructing `gross_profit`, the analysis uses a more universal
        profitability metric that was found in open sources:

        $$
        operating\\ margin =
        \\frac{operating\\ income}{total\\ revenue}
        $$
        """)

        st.divider()
        st.subheader("Research and development expenses")

        st.markdown("""
        The `research_dev` column also had many missing values. Since missing R&D expenses do not
        necessarily mean that a company spent zero on R&D, a binary indicator was created:

        - `has_rd = 1` if R&D expenses are reported,
        - `has_rd = 0` otherwise.
        """)

        st.markdown("""
        I checked the correlation of `has_rd` with the numerical fields and obtained the following results:
        """)

        corr = df_fin[
            [
                "has_rd",
                "total_revenue",
                "total_assets",
                "net_income",
                "ebitda",
                "operating_income",
                "total_liabilities",
                "cash_and_equivalents",
                "operating_cash_flow",
                "capex",
                "free_cash_flow",
                "total_equity"
            ]
        ].corr()

        corr_table = (
            corr["has_rd"]
            .drop("has_rd")
            .rename("Correlation with has_rd")
            .sort_values(key=abs, ascending=False)
            .reset_index()
            .rename(columns={"index": "Financial variable"})
        )

        corr_col1, corr_col2 = st.columns([1.2, 1])

        with corr_col1:
            st.bar_chart(
                corr_table.set_index("Financial variable")
            )

        with corr_col2:
            st.dataframe(
                corr_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Financial variable": st.column_config.TextColumn(
                        "Financial variable"
                    ),
                    "Correlation with has_rd": (
                        st.column_config.NumberColumn(
                            "Correlation",
                            format="%.3f"
                        )
                    )
                }
            )

        st.markdown("""
        Since the correlations are not high, I also checked the presence of reported R&D
        expenditure across different industry segments and obtained the following results.
        """)

        segment_map = (
            df.set_index("company_id")["segment_x"].to_dict()
        )

        company_rd = (
            df_fin
            .groupby("company_id")["has_rd"]
            .max()
            .reset_index()
        )
        company_rd["segment"] = (
            company_rd["company_id"].map(segment_map)
        )

        segment_rd = (
            company_rd
            .groupby("segment")["has_rd"]
            .mean()
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        segment_rd.plot(kind="barh", ax=ax)
        ax.set_xlabel("Share of companies reporting R&D")
        ax.set_ylabel("Business segment")
        ax.set_title("R&D Disclosure by Business Segment")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""
        The figure shows substantial differences across business segments.
        Technology-oriented segments such as SaaS, Super-apps, Social Commerce,
        and Marketplace + Gaming almost always report R&D expenditures, while
        traditional retail and marketplace businesses often do not disclose them.

        Therefore, missing values in `research_dev` were not interpreted as zero spending.
        Instead, two new variables were created:

        - `has_rd` — whether R&D expenditure is reported;
        - `rd_intensity` — R&D expenditure divided by total revenue.

        These variables preserve potentially useful information without introducing
        unverified imputations.
        """)

        st.divider()
        st.subheader("Operating cash flow")

        st.markdown("""
        Finally, missing `operating_cash_flow` values for GoTo Group were reconstructed using the
        accounting relationship between free cash flow, operating cash flow, and capital expenditure.
        Since `capex` is already stored as a negative value in the dataset, which was verified across
        the available observations, the following formula was used:
        """)

        df_check = df_fin_raw[
            df_fin_raw["operating_cash_flow"].notna()
        ].copy()

        df_check["ocf_reconstructed"] = (
            df_check["free_cash_flow"] - df_check["capex"]
        )
        df_check["ocf_diff"] = (
            df_check["ocf_reconstructed"] -
            df_check["operating_cash_flow"]
        )
        df_check["ocf_diff_pct"] = (
            df_check["ocf_diff"] /
            df_check["operating_cash_flow"].abs() * 100
        )

        st.markdown("""
        $$
        operating\\ cash\\ flow =
        free\\ cash\\ flow - capex
        $$
        """)

        st.markdown("""
        This formula was verified on companies with complete data before being used for imputation.
        """)

        ocf_check_table = (
            df_check[
                [
                    "ticker",
                    "fiscal_year",
                    "operating_cash_flow",
                    "ocf_reconstructed",
                    "ocf_diff_pct"
                ]
            ]
            .sort_values("ocf_diff_pct", key=abs)
            .head(20)
        )

        with st.expander(
            "Formula verification on complete observations",
            expanded=False
        ):
            st.dataframe(
                ocf_check_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker"),
                    "fiscal_year": st.column_config.NumberColumn(
                        "Fiscal year",
                        format="%d"
                    ),
                    "operating_cash_flow": (
                        st.column_config.NumberColumn(
                            "Reported operating cash flow",
                            format="%.2f"
                        )
                    ),
                    "ocf_reconstructed": (
                        st.column_config.NumberColumn(
                            "Reconstructed operating cash flow",
                            format="%.2f"
                        )
                    ),
                    "ocf_diff_pct": st.column_config.NumberColumn(
                        "Difference, %",
                        format="%.3f%%"
                    )
                }
            )

        st.subheader("Remaining missing values: Financial dataset")

        miss_fin = df_fin.isna().sum()
        miss_fin = miss_fin[miss_fin > 0]

        show_missing_values(
            miss_fin,
            len(df_fin),
            "No missing values remain in the cleaned financial dataset."
        )

    # ═════════════════════════════════════
    # 4.3 Macroeconomic Indicators
    # ═════════════════════════════════════

    with tab_macro:
        st.header("4.3 Macroeconomic Indicators")

        st.markdown("""
        The macroeconomic dataset was checked as a country-year panel.

        For every pair of `country_code` and `indicator_name`, the available years were compared
        with the expected continuous yearly range. This showed that most indicators were reported
        annually, including GDP, population, internet usage, mobile subscriptions, and urban
        population.

        The main exception was `account_ownership_pct_adult`. This indicator is not reported every
        year because it is based on special financial inclusion surveys. Therefore, its missing
        values were treated as structural gaps rather than data errors.
        """)

        with st.container(border=True):
            st.markdown("#### Transformation from long to wide format")
            st.markdown("""
            After this check, the macroeconomic dataset was transformed from long format into wide format:

            - each row represents one country-year pair,
            - each macroeconomic indicator becomes a separate column.

            Missing values were then filled within each country using forward fill and backward fill.
            Forward fill was used for intermediate years, while backward fill was used when the first
            available observation started in 2011 but the panel required a value for 2010.
            """)

        st.markdown("""
        For Argentina, `gdp_total_usd` was missing, but both GDP per capita and population were
        available. Therefore, total GDP was reconstructed using:

        $$
        GDP_{total} =
        GDP_{per\\ capita} \\times population
        $$
        """)

        st.subheader("Remaining missing values: Macro dataset")

        miss_macro = macro_wide.isna().sum()
        miss_macro = miss_macro[miss_macro > 0]

        show_missing_values(
            miss_macro,
            len(macro_wide),
            "No missing values remain in the cleaned macroeconomic dataset."
        )


# ══════════════════════════════════════════
# 5. BASIC PLOTS
# ══════════════════════════════════════════

elif page == "Basic Plots":
    st.title("Basic Plots")

    st.markdown("""
    This section presents simple visualizations for numerical fields from the dataset.
    The goal is to get an initial understanding of distributions, trends, and differences 
    between groups before moving to a more detailed analysis.
    """)

    # ─────────────────────────────────────
    # 1. CAGR Histogram
    # ─────────────────────────────────────
    st.subheader("1. CAGR Distribution")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(
        df_filled["cagr"].dropna(),
        bins=30,
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title("CAGR Distribution")
    axes[0].set_xlabel("CAGR")
    axes[0].set_ylabel("Number of Companies")

    sns.histplot(
        df_filled["cagr"].dropna(),
        bins=30,
        kde=True,
        ax=axes[1]
    )

    axes[1].set_title("CAGR Distribution (Clipped Outliers)")
    axes[1].set_xlabel("CAGR")
    axes[1].set_ylabel("Number of Companies")
    axes[1].set_xlim(
        df_filled["cagr"].quantile(0.02),
        df_filled["cagr"].quantile(0.98)
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
The figure shows the distribution of CAGR values across the companies in the dataset. CAGR represents the average annual rate at which a company’s stock price increased or decreased during the available observation period. The histogram shows the number of companies in each CAGR interval, while the smooth density curve shows the general shape of the distribution.
Most CAGR values are concentrated close to zero, mainly between approximately −20% and 25%. This means that most companies had moderate negative or positive average annual returns. The highest concentration is located slightly below zero, so small negative CAGR values are common in the sample. At the same time, several extreme observations can be seen near −45% and 45%. The second graph uses a narrower visible range to reduce the effect of these extreme values and make the central part of the distribution easier to examine. It shows that the distribution includes both negative and positive CAGR values and is slightly extended toward the positive side.

    """)

    # ─────────────────────────────────────
    # 2. Average Internet Usage Over Time
    # ─────────────────────────────────────
    st.subheader("2. Average Internet Usage Over Time")

    internet_yearly = (
        macro_wide
        .groupby("year")["internet_users_pct"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=internet_yearly,
        x="year",
        y="internet_users_pct",
        marker="o",
        linewidth=2,
        ax=ax
    )

    ax.set_title("Average Internet Usage Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Internet Users (%)")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
It is the average share of internet users across the countries in the dataset from 2010 to 2025. The values were grouped by year, and the mean internet usage percentage was calculated for each year. A clear upward trend can be observed between 2010 and 2024. Average internet usage increased from approximately 58% in 2010 to about 88% in 2024. The growth was gradual throughout most of the period. Faster increases can be seen between 2012 and 2016 and again between 2017 and 2020.
The highest value was reached in 2024. In 2025, the average decreased to approximately 81%. This final decline should be interpreted carefully because the 2025 data may be incomplete or may contain observations for fewer countries than the previous years. 
    """)

    # ─────────────────────────────────────
    # 3. Average Operating Margin by Fiscal Year
    # ─────────────────────────────────────
    st.subheader("3. Average Operating Margin by Fiscal Year")

    yearly = (
        df_fin
        .groupby("fiscal_year")["operating_margin"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=yearly,
        x="fiscal_year",
        y="operating_margin",
        ax=ax
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v:.1%}" for v in container.datavalues],
            padding=3
        )

    ax.set_title(
        "Average Operating Margin of E-commerce Companies by Fiscal Year",
        fontsize=13,
        pad=15
    )

    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Average Operating Margin")

    ax.set_yticklabels(
        [f"{y:.0%}" for y in ax.get_yticks()]
    )

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
A clear upward trend can be observed in rhe average operating margin. The average operating margin was −3.5% in 2022 and improved to −1.0% in 2023. In 2024, it became positive and reached 7.2%. Further growth is shown in 2025 and 2026, when the average margin increased to 8.7% and 11.0%, respectively.
Overall, the average operating margin increased by 14.5 percentage points between 2022 and 2026. The main change occurred between 2023 and 2024, when the value rose by 8.2 percentage points and moved from negative to positive. This indicates a clear improvement in the average operating results of the observations included in each fiscal year.
The results for 2025 and 2026 should be interpreted carefully if these years contain fewer company observations or incomplete financial reports. In that case, the later averages may not be directly comparable with the earlier years.

    """)

    # ─────────────────────────────────────
    # 4. Operating Margin by R&D Disclosure
    # ─────────────────────────────────────
    st.subheader("4. Operating Margin by R&D Disclosure")

    plot_df = df_fin[
        df_fin["operating_margin"].notna()
    ].copy()

    plot_df["R&D Reporting"] = plot_df["has_rd"].map({
        1: "Reports R&D",
        0: "No R&D Reported"
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.violinplot(
        data=plot_df,
        x="R&D Reporting",
        y="operating_margin",
        inner=None,
        ax=ax
    )

    sns.stripplot(
        data=plot_df,
        x="R&D Reporting",
        y="operating_margin",
        color="black",
        alpha=0.5,
        size=4,
        ax=ax
    )

    ax.set_title("Operating Margin by R&D Disclosure")
    ax.set_xlabel("")
    ax.set_ylabel("Operating Margin")
    ax.spines[["top", "right"]].set_visible(False)

    ax.set_yticklabels(
        [f"{y:.0%}" for y in ax.get_yticks()]
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
 This figure compares the distribution of operating margins for company-year observations with and without reported R&D expenditure. For companies without reported R&D, most operating margins are concentrated close to 0%, although both negative and positive values are present. The main group is located approximately between −20% and 20%. Several higher observations reach about 50%–60%, while some negative observations fall below −50%. This group has a wide central concentration and a noticeable positive upper tail.
For companies that report R&D, most observations are also concentrated around small positive and negative margins. A large part of the data is located approximately between −15% and 25%, with the highest concentration slightly above 0%. However, several strongly negative observations are visible. Two observations are located near −70%, and one extreme value is close to −270%. This observation creates the very long lower tail of the violin and strongly increases the visible range of the graph.
In both groups, many operating margins are located close to zero, so the graph does not show a clear separation between companies that report R&D and those that do not. 

    """)

    # ─────────────────────────────────────
    # 5. Revenue and Assets by Fiscal Year
    # ─────────────────────────────────────
    st.subheader("5. Average Revenue and Assets by Fiscal Year")

    yearly_metrics = (
        df_fin
        .groupby("fiscal_year")[["total_revenue", "total_assets"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.scatterplot(
        data=yearly_metrics,
        x="fiscal_year",
        y="total_revenue",
        color="green",
        s=120,
        label="Average Revenue",
        ax=ax
    )

    sns.scatterplot(
        data=yearly_metrics,
        x="fiscal_year",
        y="total_assets",
        color="royalblue",
        s=120,
        label="Average Assets",
        ax=ax
    )

    ax.plot(
        yearly_metrics["fiscal_year"],
        yearly_metrics["total_revenue"],
        color="green",
        alpha=0.6
    )

    ax.plot(
        yearly_metrics["fiscal_year"],
        yearly_metrics["total_assets"],
        color="royalblue",
        alpha=0.6
    )

    ax.set_title(
        "Average Revenue and Assets by Fiscal Year"
    )

    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Average Value (USD)")

    ax.legend()

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
The figure compares the average total revenue and average total assets of e-commerce companies by fiscal year. Both indicators are measured in US dollars. The vertical axis uses scientific notation, where 1e12 represents one trillion USD.
Average total assets were higher than average revenue in every fiscal year. In 2022, average assets reached approximately 4.4 trillion USD, while average revenue was about 0.44 trillion USD. A sharp decline in average assets was observed in 2023, when the value fell to approximately 1.9 trillion USD. Between 2023 and 2025, average assets remained relatively stable, ranging from about 1.75 to 1.9 trillion USD.
Average revenue followed a more stable upward trend from 2022 to 2025. It increased from approximately 0.44 trillion USD in 2022 to around 0.47 trillion USD in 2023, 0.51 trillion USD in 2024, and 0.58 trillion USD in 2025.
In 2026, both indicators declined sharply and became close to each other at approximately 0.1–0.15 trillion USD. This final result should be interpreted carefully because the 2026 financial data may contain fewer company observations or incomplete annual reports.

    """)

# ══════════════════════════════════════════
# 6. DETAILED OVERVIEW
# ══════════════════════════════════════════

elif page == "Detailed Overview":
    st.title("Detailed Overview")

    st.markdown("""
    This section provides a more detailed comparison-based overview of the dataset.
    The analysis focuses on differences between business segments, regions, stock-market performance,
    and macroeconomic trends.
    """)

    START_YEAR = 2015
    END_YEAR = 2024
    FORMATION_DATE = pd.Timestamp("2015-12-31")

    def prepare_equal_weighted_data(df_stick, df_ecommerce, df_filled):
        prices = df_stick.copy()
        prices["date"] = pd.to_datetime(prices["date"])

        first_dates = prices.groupby("ticker")["date"].min()
        eligible = first_dates[first_dates <= FORMATION_DATE].index

        prices = prices[prices["ticker"].isin(eligible)].copy()
        prices = prices[
            (prices["date"].dt.year >= START_YEAR) &
            (prices["date"].dt.year <= END_YEAR)
        ].copy()

        prices = prices.sort_values(["ticker", "date"])

        prices["normalized"] = (
            prices["adj_close"] /
            prices.groupby("ticker")["adj_close"].transform("first") *
            100
        )

        price_wide = (
            prices
            .pivot(index="date", columns="ticker", values="normalized")
            .ffill()
        )

        equal_weight = (
            price_wide
            .mean(axis=1)
            .reset_index(name="normalized")
        )

        coverage = (
            prices
            .groupby("date")["ticker"]
            .nunique()
            .reset_index(name="n_companies")
        )

        ecom_compare = df_ecommerce.copy()
        ecom_compare["date"] = pd.to_datetime(ecom_compare["date"])

        ecom_compare = ecom_compare[
            (ecom_compare["date"].dt.year >= START_YEAR) &
            (ecom_compare["date"].dt.year <= END_YEAR)
        ].copy()

        ecom_compare["norm"] = (
            ecom_compare["index_level"] /
            ecom_compare["index_level"].iloc[0] *
            100
        )

        region_map = (
            df_filled[["ticker_x", "region_x"]]
            .drop_duplicates()
            .rename(columns={"ticker_x": "ticker"})
        )

        prices_region = prices.merge(
            region_map,
            on="ticker",
            how="left"
        )

        return prices, equal_weight, ecom_compare, coverage, eligible, prices_region


    def prepare_yearly_cagr(df_stick, df_filled):
        prices = df_stick.copy()
        prices["date"] = pd.to_datetime(prices["date"])
        prices["year"] = prices["date"].dt.year

        prices = prices[
            (prices["year"] >= START_YEAR) &
            (prices["year"] <= END_YEAR)
        ].copy()

        yearly = (
            prices
            .sort_values(["ticker", "date"])
            .groupby(["ticker", "year"])
            .last()
            .reset_index()
        )

        first_price = yearly.groupby("ticker")["adj_close"].first()
        first_year = yearly.groupby("ticker")["year"].first()

        yearly["first_price"] = yearly["ticker"].map(first_price)
        yearly["first_year"] = yearly["ticker"].map(first_year)

        yearly["years_elapsed"] = yearly["year"] - yearly["first_year"]
        yearly = yearly[yearly["years_elapsed"] > 0].copy()

        yearly["cagr"] = (
            (yearly["adj_close"] / yearly["first_price"]) **
            (1 / yearly["years_elapsed"]) - 1
        )

        region_map = (
            df_filled[["ticker_x", "region_x"]]
            .drop_duplicates()
            .rename(columns={"ticker_x": "ticker"})
        )

        yearly = yearly.merge(
            region_map,
            on="ticker",
            how="left"
        )

        return yearly


    prices, equal_weight, ecom_compare, coverage, eligible, prices_region = (
        prepare_equal_weighted_data(df_stick, df_ecommerce, df_filled)
    )

    yearly = prepare_yearly_cagr(df_stick, df_filled)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "CAGR by Segment",
        "E-commerce vs Macro",
        "Portfolio vs Index",
        "Portfolio by Region",
        "CAGR by Region"
    ])

    # ─────────────────────────────────────
    # TAB 1: CAGR by Segment
    # ─────────────────────────────────────

    with tab1:
        st.subheader("CAGR Distribution by Business Segment")

        st.markdown("""
        This plot compares the distribution of long-term stock growth across the five most common
        business segments in the dataset.
        """)

        top_segments = (
            df_filled["segment_x"]
            .value_counts()
            .head(5)
            .index
        )

        plot_df = df_filled[
            df_filled["segment_x"].isin(top_segments)
        ].copy()

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.kdeplot(
            data=plot_df,
            x="cagr",
            hue="segment_x",
            fill=True,
            alpha=0.25,
            common_norm=False,
            ax=ax
        )

        ax.set_title("CAGR Distribution by Business Segment")
        ax.set_xlabel("CAGR")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
CAGR Distribution by Business Segment
                    

The figure compares the distribution of CAGR values across five business segments:

                    
•	Marketplace + Cloud;
•	Fashion;
•	Food Delivery;
•	Marketplace + Fintech;
•	Payments.
                    

Logic Used to Construct the Figure
                    

The graph was constructed from the company-level CAGR values and the business segment variable segment_x.
First, companies were separated into groups according to their business segment. The CAGR values were then collected separately for each group. After that, a kernel density estimate was created for every selected segment. Kernel density estimation produces a smooth curve from the original observations. It is similar to a smoothed histogram, but separate bars are replaced by a continuous line.
Each segment was shown with its own curve and colour. The areas under the curves were filled with partial transparency. This made it possible to compare several distributions on the same figure and to see where they overlap. The same horizontal scale was used for all five segments. Therefore, their positions, ranges, peaks, and tails can be compared directly.
The graph should be interpreted through several features:
                    

•	the position of the peak;
•	the width of the curve;
•	the height of the curve;
•	the length of the left and right tails;
•	the overlap between different segments.
                    

The peak shows the CAGR range where the values are most concentrated. The width shows how different the observations are within the segment. The tails show whether some values are located far from the main part of the distribution.
General Pattern
The five segments have visibly different CAGR distributions.
                    

•	the fashion segment is concentrated mainly in the negative CAGR area. Its curve is narrow and has the highest peak
•	the marketplace plus cloud segment is centred in the positive CAGR area. Its curve is wider and extends far into positive values
•	the Food Delivery segment is concentrated close to zero and slightly above zero. Its distribution is narrower than the distributions of Marketplace + Cloud, Marketplace + Fintech, and Payments
•	the Marketplace + Fintech segment has the widest visible range. Its curve extends far into both negative and positive CAGR values
•	the Payments segment is centred close to zero. Its distribution overlaps strongly with Food Delivery, Marketplace + Cloud, and Marketplace + Fintech.

                    
                    These differences show that the CAGR values are not distributed in the same way across all business segments.
Fashion
The Fashion segment is represented by the orange curve.
Its highest density is located at approximately −0.14, which corresponds to a CAGR of about −14%. This is the most common area of the distribution. The curve is mainly located between approximately −0.30 and 0.05. Most of the visible density is found in the negative part of the horizontal axis. The peak is much higher than the peaks of the other segments. It reaches a density of approximately 4.6. This does not mean that the Fashion segment contains the largest number of companies. It means that its CAGR values are strongly concentrated within a relatively narrow interval. The left side of the curve begins near approximately −0.35. Density then increases quickly as the curve approaches the main peak.
After the peak near −14%, density decreases sharply. The curve approaches zero at a CAGR slightly above 0.05, or approximately 5%. The distribution is therefore narrow in comparison with most other segments. The observed CAGR values appear to be less spread out. The curve is also clearly shifted to the left of zero. This means that the main concentration of Fashion observations is located in the negative CAGR range.Only a small part of the curve is located above zero. Positive CAGR values appear to be less common in this segment than negative values.
Among all five segments, Fashion is the only segment whose main peak is clearly located below zero.
                    Food Delivery
                    

The Food Delivery segment is represented by the green curve.
Its peak is located slightly above zero, at approximately 0.05. This corresponds to a CAGR of around 5%. The maximum density is approximately 2.5. The peak is lower than the Fashion peak, but higher than the peaks of Marketplace + Cloud, Marketplace + Fintech, and Payments.
The visible distribution begins near approximately −0.30 and ends near approximately 0.4 Most of the density is located between approximately −0.15 and 0.30. This means that the central part of the distribution includes both negative and positive CAGR values. However, the highest part of the curve is located above zero. Positive values are more strongly represented around the main peak. The left side of the curve rises gradually from the negative CAGR area. The curve reaches its maximum near 5% and then decreases toward the positive side. The right tail extends farther from zero than the main negative part of the distribution. The curve remains visible until approximately 45%–50%.
The distribution is wider than the Fashion distribution. Therefore, Food Delivery CAGR values appear to be more varied.
At the same time, it is narrower than the Marketplace + Fintech distribution and slightly narrower than the Marketplace + Cloud and Payments distributions.

                    
                    Marketplace + Cloud


The Marketplace + Cloud segment is represented by the blue curve.
Its peak is located at approximately 0.16 to 0.18. This corresponds to a CAGR of around 16%–18%. This is the most positive peak among the five displayed distributions. The maximum density is approximately 1.8. The curve is lower than the curves for Fashion and Food Delivery. However, it is spread across a wider CAGR interval. The distribution begins near approximately −0.30 and extends to approximately 0.65. The central part of the curve is located mainly in the positive CAGR range. The highest density is observed between approximately 0.10 and 0.25. The curve crosses the zero point with a relatively high density. Therefore, values close to zero are also present. The negative tail is visible, but it is shorter than the positive tail. The curve extends farther to the right. The right side remains visible beyond 0.40 and approaches zero only near approximately 0.60–0.65. This indicates that the Marketplace + Cloud distribution contains a wide range of positive CAGR values.
The curve is wider than the Fashion and Food Delivery curves. This shows that CAGR values within Marketplace + Cloud are more spread out.
The distribution also overlaps strongly with Food Delivery, Payments, and Marketplace + Fintech, especially between approximately 0.00 and 0.35.

                    Marketplace + Fintech

The Marketplace + Fintech segment is represented by the red curve.
Its peak is located at approximately 0.10, or about 10% CAGR. The maximum density is close to 1.5. This is one of the lowest peaks on the graph. The lower peak is connected with the large width of the curve. The observations are spread over a very wide interval instead of being concentrated around one narrow range. The visible distribution begins near approximately −0.65 and extends to approximately 0.80.
This is the widest range among all five segments. The curve contains the longest negative tail. It remains visible far below −0.40 and approaches zero only near approximately −0.60. The right tail is also the longest. It remains visible above 0.60 and reaches approximately 0.80.
The main part of the curve is located around positive CAGR values, but a substantial part of the distribution is also located below zero. The curve rises gradually from the far negative area. It reaches its maximum near 10% and then decreases gradually toward the positive tail.
The left and right sides are both broad. This shows that the Marketplace + Fintech segment contains highly different CAGR values.
Payments

                    The Payments segment is represented by the purple curve.

Its peak is located close to zero, approximately between 0.00 and 0.03. This corresponds to a CAGR between 0% and 3%. The maximum density is slightly above 2.0. The distribution begins near approximately −0.45 and extends to approximately 0.55. The central part of the curve covers both negative and positive CAGR values. The highest concentration is found close to zero.
The curve rises from the negative CAGR area and reaches its maximum near zero. It then decreases gradually toward the positive side. The positive tail extends slightly farther than the negative tail. The Payments curve is wider than the Fashion and Food Delivery curves. It is also narrower than the Marketplace + Fintech distribution.A strong overlap can be seen between Payments and Food Delivery. Their curves are close between approximately −0.10 and 0.25. Payments also overlaps with Marketplace + Cloud. However, the Marketplace + Cloud peak is located farther to the right. The Payments distribution is therefore centred closer to zero than Marketplace + Cloud and Marketplace + Fintech.
Some conclusion
First, the Fashion distribution is shifted toward negative CAGR values. Its peak is approximately −14%, and its range is relatively narrow.
Second, the Marketplace + Cloud distribution is shifted toward positive CAGR values. Its peak is approximately 16%–18%, and its positive tail extends beyond 60%.
Third, the Food Delivery distribution is centred slightly above zero. Its peak is approximately 5%.
Fourth, the Payments distribution is centred close to zero and covers both negative and positive values.
Fifth, the Marketplace + Fintech segment has the widest distribution. It extends from approximately −60% to 80%.
Moreover, the graph shows smoothed distributions rather than individual observations and the figure provides a visual comparison of the location, concentration, spread, and overlap of CAGR values across business segments. It does not explain why these differences exist. It only shows how the observed values are distributed in the dataset.

        """)

    # ─────────────────────────────────────
    # TAB 2: E-commerce vs Macro
    # ─────────────────────────────────────

    with tab2:
        st.subheader("E-commerce Growth and Macroeconomic Trends")

        st.markdown("""
        Here we compare the normalized e-commerce index with several macroeconomic indicators.
        Both series are rebased to 100 in 2015, which makes their growth dynamics easier to compare.
        """)

        ecom = df_ecommerce.copy()
        ecom["date"] = pd.to_datetime(ecom["date"])

        ecom = ecom[
            ecom["date"].dt.year >= START_YEAR
        ].copy()

        ecom_yearly = (
            ecom
            .assign(year=ecom["date"].dt.year)
            .groupby("year")["index_level"]
            .mean()
            .reset_index()
        )

        ecom_yearly = ecom_yearly[
            (ecom_yearly["year"] >= START_YEAR) &
            (ecom_yearly["year"] <= END_YEAR)
        ].copy()

        ecom_yearly["date"] = pd.to_datetime(
            ecom_yearly["year"].astype(str) + "-06-01"
        )

        ecom_yearly["ecom_norm"] = (
            ecom_yearly["index_level"] /
            ecom_yearly["index_level"].iloc[0] *
            100
        )

        indicators = {
            "internet_users_pct": "Internet users (%)",
            "account_ownership_pct_adult": "Account ownership (%)",
            "mobile_subs_per_100": "Mobile subscriptions",
            "gdp_per_capita_usd": "GDP per capita",
            "fdi_inflow_pct_gdp": "FDI inflow (% GDP)",
            "urban_population_pct": "Urban population (%)",
        }

        fig = plt.figure(figsize=(18, 10))
        gs_obj = GridSpec(2, 3, figure=fig)

        axes = []
        for i in range(2):
            for j in range(3):
                axes.append(fig.add_subplot(gs_obj[i, j]))

        for ax, (indicator, title) in zip(axes, indicators.items()):
            ax.plot(
                ecom_yearly["date"],
                ecom_yearly["ecom_norm"],
                color="steelblue",
                linewidth=2,
                label="E-commerce Index"
            )

            ax.set_ylabel("E-commerce (2015=100)", color="steelblue")
            ax.tick_params(axis="y", labelcolor="steelblue")

            macro = (
                macro_wide[
                    (macro_wide["year"] >= START_YEAR) &
                    (macro_wide["year"] <= END_YEAR)
                ]
                .groupby("year")[indicator]
                .median()
                .reset_index()
                .dropna()
            )

            if not macro.empty and 2015 in macro["year"].values:
                macro["date"] = pd.to_datetime(
                    macro["year"].astype(int).astype(str) + "-06-01"
                )

                base_value = macro.loc[
                    macro["year"] == START_YEAR,
                    indicator
                ].iloc[0]

                macro["macro_norm"] = macro[indicator] / base_value * 100

                ax2 = ax.twinx()

                ax2.plot(
                    macro["date"],
                    macro["macro_norm"],
                    color="coral",
                    linewidth=2,
                    linestyle="--",
                    label=title
                )

                ax2.set_ylabel(f"{title} (2015=100)", color="coral")
                ax2.tick_params(axis="y", labelcolor="coral")

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()

                ax.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    fontsize=8,
                    loc="upper left"
                )

            ax.set_title(title)
            ax.grid(alpha=0.3)

        fig.suptitle(
            "E-commerce Growth and Median Macroeconomic Trends",
            fontsize=16,
            y=0.98
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""

                    E-commerce Growth and Median Macroeconomic Trends
The figure compares the development of the e-commerce index with six macroeconomic indicators between 2015 and 2024.
The following indicators are shown:
                    

	internet users;
	account ownership;
	mobile subscriptions;
	GDP per capita;
	foreign direct investment inflows;
	urban population.
                    

The blue line represents the e-commerce index in every subplot. The orange dashed line represents one macroeconomic indicator.
Method Used to Construct the Figure
                    
Several data transformation steps were used before the figure was created.
                    
First, the e-commerce data was aggregated by year. The original index contained more frequent observations, so the annual mean was calculated for every year.
Second, the macroeconomic indicators were grouped by year. Since the dataset contains several countries, the median value across countries was calculated for each indicator and year.
The median was used instead of the mean because it is less affected by countries with extremely large or extremely small values. Therefore, the orange line represents the general central trend across the countries included in the dataset.
After the annual values were calculated, every series was rebased to 2015.

The following formula was used:
                    
Index in year t = (Value in year t / Value in 2015) × 100
                    

As a result, the value for 2015 is equal to 100 for every series.
                    
A value above 100 means that the indicator increased compared with 2015. For example, a value of 120 means that the indicator is approximately 20% higher than its 2015 level.
A value below 100 means that the indicator decreased compared with 2015.
This transformation was required because the indicators are measured in different units. GDP per capita is measured in money, internet usage is measured as a percentage, and mobile subscriptions are measured using another scale. After rebasing, their relative changes can be compared on the same type of scale.
The same e-commerce index is shown in every subplot.

                    General Trend of the E-commerce Index
The index starts at 100 in 2015. A very small increase is observed in 2016.
Between 2016 and 2019, the index increases steadily. It rises from approximately 100 to around 190 in 2018 and then reaches approximately 220 in 2019. The increase becomes much stronger after 2019. The index reaches approximately 350 in 2020. The highest point is observed in 2021, when the index reaches approximately 540. This means that the e-commerce index was more than five times higher than its 2015 level. After the 2021 peak, a sharp decline is observed. The index falls to approximately 240 in 2022.
A further smaller decline is observed in 2023, when the index reaches approximately 210. In 2024, the index increases again to approximately 255. However, it remains far below the 2021 peak.

Therefore, the general e-commerce pattern can be divided into four stages:
	slow growth from 2015 to 2017;
	stronger growth from 2018 to 2020;
	a sharp peak in 2021;
	a decline in 2022–2023 and a partial recovery in 2024.

                    Internet Users
The first subplot compares the e-commerce index with the median share of internet users.
The internet users indicator shows a mainly upward trend. It starts at 100 in 2015. A strong increase is observed in 2016, when the index reaches approximately 113. The indicator continues to rise gradually between 2016 and 2020. It reaches approximately 120 by 2020. A small decline is observed in 2021. The value falls from approximately 120 to around 117. After that, growth continues. The indicator increases in 2022, 2023, and 2024. 
The highest value is observed in 2024, at approximately 123. This means that the median internet usage level was around 23% higher than in 2015..
Both indicators generally increase between 2015 and 2020. However, their movements are not identical. The e-commerce index reaches its highest value in 2021, while the internet users indicator decreases slightly in the same year. After 2021, the e-commerce index falls strongly, while internet usage continues to increase.
Therefore, the two series have a long-term upward direction before 2021, but the internet users indicator does not follow the sharp changes in the e-commerce index.

                    Account Ownership
The second subplot compares the e-commerce index with the median account ownership indicator.
The account ownership series changes only slightly during the observed period. It starts at 100 in 2015 and remains at the same level in 2016. A small decline is observed in 2017. The value falls to approximately 99.5 and remains at this level until 2020. A clear increase is observed in 2021. The indicator rises to approximately 102.2. It then remains almost unchanged between 2021 and 2023. Another increase is observed in 2024, when the value reaches approximately 103.5.
The step-like shape of the line should be noted. Long horizontal sections are visible because account ownership is not reported every year in the original data. Missing intermediate observations were filled during the preparation of the country-year panel.
The total change in account ownership is small. By 2024, the indicator is only around 3.5% higher than in 2015. This differs strongly from the e-commerce index, which shows much larger changes. The sharp growth and decline of the e-commerce index are not repeated by the account ownership series.
The only visible similarity is that both indicators increase in 2021. However, the size of the change is very different. Overall, account ownership remains relatively stable, while the e-commerce index changes strongly over time.

                    Mobile Subscriptions
The third subplot compares the e-commerce index with median mobile subscriptions.
.It starts at 100 in 2015 and decreases slightly in 2016.A rise is observed in 2017, when the value reaches approximately 102. The indicator then falls sharply in 2018 to approximately 98, which is the lowest point in the series. A strong recovery is observed in 2019. The value increases to approximately 103. Another decline occurs in 2020, when the indicator returns to around 100.
The indicator rises strongly again in 2021 and reaches approximately 105. A decline is observed in 2022, followed by renewed growth in 2023 and 2024.
The highest value is reached in 2024, at approximately 106.5.
Unlike internet usage, mobile subscriptions do not follow a continuous upward path. The series contains several short-term changes in direction. The e-commerce index and mobile subscriptions both increase strongly in 2021. However, similar movement is not observed during all other years. For example, mobile subscriptions decrease in 2018, while the e-commerce index continues to increase. In 2022, both series decline. After that, the e-commerce index decreases again in 2023, while mobile subscriptions begin to recover. Therefore, several periods of similar movement are visible, but the complete trends are different.

                    GDP per Capita
GDP per capita starts at 100 in 2015 and increases slightly in 2016. A stronger increase is observed between 2016 and 2018. The indicator reaches approximately 117 in 2018. After that, it declines in 2019 and again slightly in 2020. However, the value remains above its 2015 level. A sharp increase is observed in 2021. The indicator reaches approximately 124, which is the highest point in the series. GDP per capita then falls in 2022, 2023, and 2024. The 2024 value is approximately 110, which remains around 10% above the 2015 level. The e-commerce index and GDP per capita both reach their highest point in 2021.
Both series also decline sharply in 2022. This is the clearest period of similar movement in the figure. However, the earlier years show a less direct relationship. GDP per capita rises strongly in 2017–2018, while the e-commerce index increases more gradually.In 2024, the e-commerce index begins to recover, while GDP per capita continues to decline.
FDI Inflow
A strong decline is observed in 2016. The value falls to approximately 90. A small increase follows in 2017, but the general downward movement continues in 2018 and 2019. The indicator reaches approximately 83–85 by 2019 and 2020. A strong recovery is observed in 2021. The value rises to approximately 95. The level remains close to 95 in 2022. After that, a sharp decline is observed. The indicator falls to approximately 74 in 2023 and then to approximately 48 in 2024.
The e-commerce index and FDI do not follow the same general trend during most of the period. Between 2015 and 2020, the e-commerce index increases, while FDI generally decreases. Both indicators increase in 2021. However, after 2022, FDI continues to fall strongly.The e-commerce index also declines in 2022 and 2023, but it starts to recover in 2024. FDI continues to decrease in 2024.
Urban Population
It starts at 100 in 2015 and increases slowly in 2016, 2017, and 2018. A small decline is observed between 2018 and 2020. The highest level is reached in 2024, at approximately 101.
This means that the total increase over the entire period is only around 1%.
The urban population series changes very slowly compared with the e-commerce index. Both indicators increase after 2020 and move upward in 2021. However, the e-commerce index falls sharply after 2021, while the urban population indicator continues to increase. The urban population indicator shows a stable long-term increase, while the e-commerce index shows a much more volatile pattern.
The overall pattern shows that the e-commerce index experienced a strong growth period followed by a major decline and a partial recovery. Most macroeconomic indicators changed more gradually. The closest visual similarity is observed between the e-commerce index and GDP per capita around the 2021 peak and the 2022 decline. Internet users and urban population show long-term growth, but they do not reproduce the sharp fall in the e-commerce index. Mobile subscriptions show several short-term movements in the same direction as the e-commerce index, but no stable common trend is visible.
Overall, the figure suggests that some indicators move in the same direction as the e-commerce index during individual years. However, no macroeconomic series follows the complete e-commerce trend across the whole period.

        """)

    # ─────────────────────────────────────
    # TAB 3: Portfolio vs E-commerce Index
    # ─────────────────────────────────────

    with tab3:
        st.subheader("Equal-Weighted Portfolio vs E-commerce Index")

        col1, col2, col3 = st.columns(3)

        col1.metric("Eligible companies", len(eligible))
        col2.metric("Min companies per day", int(coverage["n_companies"].min()))
        col3.metric("Max companies per day", int(coverage["n_companies"].max()))

        st.markdown("""
        This figure compares a hypothetical equal-weighted investment portfolio of e-commerce
        companies with the general e-commerce index. The portfolio is rebalanced conceptually by
        taking the average normalized stock price across eligible companies.
        """)

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            equal_weight["date"],
            equal_weight["normalized"],
            linewidth=2.5,
            color="steelblue",
            label=f"Equal-weighted Portfolio ({len(eligible)} companies)"
        )

        ax.plot(
            ecom_compare["date"],
            ecom_compare["norm"],
            linewidth=2,
            linestyle="--",
            color="coral",
            label="E-commerce Index"
        )

        ax.axvspan(
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2021-12-31"),
            alpha=0.15,
            color="gray"
        )

        ax.text(
            pd.Timestamp("2020-04-01"),
            equal_weight["normalized"].max() * 0.95,
            "COVID-19",
            fontsize=10
        )

        ax.set_title(
            "Growth of a $1 Equal-Weighted Investment in E-commerce Companies",
            fontsize=14
        )

        ax.set_ylabel("Index Value (2015 = 100)")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Show daily company coverage"):
            st.dataframe(coverage.head(20), use_container_width=True)

        st.markdown("""
The figure compares an equal-weighted portfolio of 19 e-commerce companies with the general e-commerce index. Both series were rebased to 2015 = 100, so their relative growth can be compared.

                    Period before 2020
From 2015 to 2016, both series remained close to their initial level of 100 and showed only small changes.
Starting from 2017, both lines began to grow. The equal-weighted portfolio increased faster than the e-commerce index. By the end of 2019, the portfolio had reached approximately 350–370, while the index was close to 240–250.
Both series experienced several short declines during this period, especially around 2018 and early 2019. However, the general trend before 2020 was positive.

                    from 2020 to 2024
At the beginning of 2020, both series declined. This fall was followed by very strong growth during the shaded COVID-19 period. The equal-weighted portfolio increased much faster. It rose from approximately 300–400 in early 2020 to more than 1,000 in 2021. The e-commerce index also increased, but its peak was lower, at approximately 600–650.
Both series reached their highest levels during 2021. After that, a strong decline was observed. During 2022, the portfolio fell to approximately 300–400, while the index declined to around 200–250. The decrease continued into early 2023. From 2023 to 2024, both series started to recover. The portfolio generally remained above the e-commerce index and showed stronger growth.

                    after 2024
During the second half of 2024 and the beginning of 2025, both series continued to increase. By the end of the observed period, both series remained above their 2015 levels. However, neither series returned to its 2021 peak.
The two series generally moved in the same direction. Both showed gradual growth before 2020, rapid growth during 2020–2021, a strong decline during 2022, and a recovery after 2023.

        """)

    # ─────────────────────────────────────
    # TAB 4: Portfolio by Region
    # ─────────────────────────────────────

    with tab4:
        st.subheader("Equal-Weighted Portfolio Growth by Region")

        st.markdown("""
        This graph compares equal-weighted stock performance across regions. Each regional index is
        calculated as the average normalized stock price of companies from that region.
        """)

        regional_indices = []

        for region, group in prices_region.dropna(subset=["region_x"]).groupby("region_x"):
            wide = (
                group
                .pivot(index="date", columns="ticker", values="normalized")
                .ffill()
            )

            idx = wide.mean(axis=1)

            regional_indices.append(
                pd.DataFrame({
                    "date": idx.index,
                    "index": idx.values,
                    "region": region
                })
            )

        regional_indices = pd.concat(regional_indices, ignore_index=True)

        fig, ax = plt.subplots(figsize=(14, 6))

        for region, group in regional_indices.groupby("region"):
            ax.plot(
                group["date"],
                group["index"],
                linewidth=2,
                label=region
            )

        ax.set_title(
            "Growth of a $1 Equal-Weighted Investment by Region",
            fontsize=14
        )

        ax.set_ylabel("Index Value (2015 = 100)")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        ax.legend(title="Region")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""

                    Growth of a $1 Equal-Weighted Investment by Region
The figure compares the performance of equal-weighted portfolios of e-commerce companies from six regions:
                   
•	Asia Pacific;
                    
•	Europe;
                    
•	Greater China;
                    
•	India;
                    
•	Latin America;
                    
•	North America.
                    
                    
All regional portfolios were rebased to 2015 = 100. Therefore, the lines show how the value of the same initial investment changed over time.

                    How the Regional Portfolios Were Calculated
First, the stock-price history of every company was converted into a normalized growth series.
For each company, the adjusted closing price on every date was divided by its adjusted closing price at the beginning of the period:
Company growth at time t = Adjusted closing price at time t / Adjusted closing price at the beginning
A value of 1 means that the stock price remained at its initial level. A value of 2 means that it became twice as high, while a value of 0.5 means that it fell to half of its initial level.
After that, companies were grouped by region. 

The same weight was assigned to every company within each regional group. 
This means that a hypothetical 1dollar investment was divided equally between all available companies in the region.
For example, when a region contained five companies, 0.20dollars was assigned to each company.

The regional portfolio value was calculated as the average of the normalized company growth values:
Regional portfolio value at time t = Sum of company growth values / Number of companies in the region
                    
Finally, the result was multiplied by 100 so that every regional portfolio started from 100 in 2015:
Regional index at time t = Regional portfolio value at time t × 100
So
                    
•	an index value of 100 corresponds to the original 1dollar investment;
                    
•	an index value of 200 corresponds to approximately 2dollar;
                    
•	an index value of 500 corresponds to approximately 5dollar;
                    
•	an index value of 1,000 corresponds to approximately 10dollar;
                    
•	an index value of 2,000 corresponds to approximately 20dollar.
                    

                    Period Before 2020
Between 2015 and 2016, all regional portfolios remained relatively close to 
their initial value of 100. Only small increases and decreases were observed. 
    Starting from 2017, the regional trends became more different.
North America showed the strongest growth before 2020. Its index increased from around 100 in 2015 to approximately 600–650 by 2019. 
A 1 dollar investment would therefore have grown to approximately 6–6.50 dollars.
Latin America also increased strongly. By 2019, its index was close to 400–500, 
which corresponds to a portfolio value of approximately 4–5 dolars.
Europe showed moderate growth. Its index increased to approximately 200–250 by 2019.
India also followed an upward trend and reached approximately 250 by the end of 2019.
Greater China grew more slowly and remained mostly between approximately 100 and 150 before 2020.
Asia Pacific showed the weakest performance during this period. Its index generally remained below 100 
after 2016 and was close to approximately 40–60 by 2019.

                    Period From 2020 to 2024
The largest differences between regions appeared during this period.
North America increased very strongly during 2020 and 2021. Its index rose above 2,000 
and reached a maximum of approximately 2,200–2,300.
This means that the original $1 investment would have been worth approximately 22–23 dollars at the highest point.
However, a sharp decline was observed in 2022. The North American index fell to approximately 500–700. 
A partial recovery followed during 2023 and 2024, and the index reached approximately 900–1,000 during parts of 2024.
Latin America also showed strong growth during 2020 and 2021. Its index increased above 1,500 in 2021. 
It then declined to approximately 700–900 during 2022.
India increased more gradually. Its index reached approximately 600 in 2021 and around 800 in late 2021.
 A decline followed in 2022, when the value fell to approximately 450–550. After that, a new upward trend was observed. The Indian portfolio increased through 2023 and 2024 and approached approximately 900–1,000 by the end of 2024.
Europe reached its highest value of approximately 400 during 2021. After that, the European portfolio followed a mainly downward trend. Its index fell to approximately 150 in 2022 and remained close to 100–150 during 2023 and 2024. This means that most of the earlier growth had been lost by the end of the period.
Greater China also reached a local maximum during 2021, at approximately 250–300. It then declined during 2022. From 2022 to 2024, the index generally remained close to approximately 100–150. Therefore, the final value was only slightly above the original 2015 level.
Asia Pacific remained the lowest regional portfolio during most of the period. Its index was generally below 100 and often close to approximately 40–70.

                    After 2024
At the end of 2024 and the beginning of 2025, different regional movements were observed.
Latin America remained the highest regional portfolio for most of this final period. 
However, a decline was observed near the beginning of 2025. Its index fell 
from approximately 1,650–1,700 to around 1,350–1,450.
North America increased sharply near the end of 2024 and briefly reached approximately 1,400–1,500. 
A small decline followed, and the final value was close to approximately 1,350–1,400.
India continued its upward trend and reached approximately 1,050 by the beginning of 2025.
                     This corresponds to a value of approximately $10.50 for the original $1 investment.
Europe, Greater China, and Asia Pacific remained much lower.
Europe finished close to approximately 100, which is close to the initial $1 value.
Greater China also ended close to approximately 100–120.
Asia Pacific finished below its initial level, at approximately 50–60. 
This means that the original 1 dollar investment would have been worth only around 0.50–0.60.

        """)

    # ─────────────────────────────────────
    # TAB 5: CAGR by Region
    # ─────────────────────────────────────

    with tab5:
        st.subheader("CAGR Distribution by Region")

        st.markdown("""
        We analysed the distribution of annualized stock growth rates across all company-year
        observations between 2015 and 2024. Rather than focusing only on the final performance of
        each company, this approach incorporates intermediate stages of firm development throughout
        the sample period.
        """)

        regions = yearly["region_x"].dropna().unique()
        n = len(regions)

        overall_median = yearly["cagr"].median()

        fig = plt.figure(figsize=(14, 3 * n))
        gs_obj = GridSpec(n, 1, hspace=0.6)

        for i, region in enumerate(regions):
            ax = fig.add_subplot(gs_obj[i])

            data = yearly.loc[
                yearly["region_x"] == region,
                "cagr"
            ].dropna()

            if len(data) < 3:
                ax.set_title(f"{region} (n={len(data)}) — too few observations")
                ax.axis("off")
                continue

            low, high = data.quantile(0.05), data.quantile(0.95)
            data_clipped = data.clip(low, high)

            sns.kdeplot(
                data_clipped,
                ax=ax,
                fill=True,
                alpha=0.3,
                linewidth=2
            )

            region_median = data.median()

            ax.axvline(
                region_median,
                color="steelblue",
                linestyle="--",
                linewidth=1.5,
                label=f"Median = {region_median:.2f}"
            )

            ax.axvline(
                overall_median,
                color="red",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                label=f"Overall = {overall_median:.2f}"
            )

            ax.axvline(
                0,
                color="black",
                linestyle="-",
                linewidth=0.8,
                alpha=0.3
            )

            ax.set_title(
                f"{region} (n={len(data)})",
                fontsize=11,
                fontweight="bold"
            )

            ax.set_xlabel("CAGR" if i == n - 1 else "")
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_xlim(-1, 1)

        fig.suptitle(
            "CAGR Distribution by Region",
            fontsize=14,
            y=1.01,
            fontweight="bold"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
The figure was created by grouping the company observations by region and examining the distribution of CAGR values separately for each group. Kernel density estimation was used to produce a smooth distribution curve for every region. This method is similar to a histogram, but the individual bars are replaced by a continuous curve, which makes the general shape, concentration, spread, and tails of the distribution easier to compare. The horizontal axis shows CAGR values, while the vertical axis shows density rather than the number of observations. The number of observations used for each region is shown in the title of every subplot. A blue dashed line marks the median CAGR for the selected region, and a red dotted line marks the overall median CAGR for the full dataset, which is equal to 0.11, or 11%. The same horizontal range from −1 to 1 was used for all regions so that the positions and widths of the distributions could be compared directly. This means that values on the left side of zero represent negative CAGR, while values on the right side represent positive CAGR. The regional observations were not combined into one average value, because the purpose of the graph was to show the full distribution inside each region rather than only one summary statistic.
Asia Pacific contains 21 observations and has a median CAGR of −0.09, or −9%, which is 20 percentage points below the overall median of 11%. Its curve has the highest concentration slightly below zero, while a long positive tail extends toward values close to 1.00. This means that most observations are concentrated around small negative CAGR values, but several much higher positive values are also present. 
North America contains 95 observations and has a median CAGR of 0.25, or 25%, which is 14 percentage points above the overall median. Its distribution is wide and mainly concentrated in the positive part of the graph, although negative observations are also visible. The curve extends far to the right, which shows that a number of observations have high positive CAGR values.
Europe contains 58 observations and has a median of 0.01, or 1%, which is 10 percentage points below the overall median. The European curve has a clear peak close to zero and a smaller positive section extending toward approximately 0.70, so most values are concentrated around low or slightly negative growth, while some stronger positive observations are also present.
China contains 33 observations and has a median CAGR of 0.07, or 7%, which is 4 percentage points below the overall median. Its main peak is located close to zero, but most of the curve extends into the positive area, with the right tail reaching approximately 0.70. 
India contains 35 observations and has a median CAGR of 0.17, or 17%, which is 6 percentage points above the overall median. Its distribution is very wide and includes a long negative tail reaching close to −1.00, while the highest density is located around 0.20–0.25. This shows that the Indian observations include both strongly negative and strongly positive CAGR values. 
Latin America contains the smallest sample, with 15 observations, and has the highest regional median of 0.37, or 37%. This is 26 percentage points above the overall median. Its distribution is shifted furthest to the right, with the highest concentration around 0.40–0.45. However, the curve is also wide and includes negative values. 

        """)

# 7. HYPOTHESIS TESTING

elif page == "Hypothesis Testing":
    st.title("Hypothesis Testing")

    tab, tab_h1, tab_h2 = st.tabs([
        "Data transformation",
        "H1: Revenue Growth vs Stock Return",
        "H2: Volatility Developed vs Emerging"
    ])
    with tab:
        st.header("Data Transformation")

        st.markdown("""
        Before the hypotheses were tested, the original tables were transformed into
        several analytical datasets.
        """)

        # Краткое визуальное резюме
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Company-year observations",
                value="121"
            )

        with col2:
            st.metric(
                label="Companies",
                value="43"
            )

        with col3:
            st.metric(
                label="Final analytical datasets",
                value="2"
            )

        st.divider()
        st.subheader("1. Company and Market Data Preparation")

        with st.container(border=True):
            st.markdown("""
            The company description table was first merged with the company-level
            market metrics by `company_id`.

            All date columns used in the calculations were converted to the datetime
            format. This included daily stock-price dates, `history_start`,
            `history_end`, and `founded_date`.

            Several foundation dates contained `00` instead of a known month or day.
            These unknown parts were replaced with `01` so that the values could be
            stored as valid dates.

            The daily stock-price observations were then sorted by company and date.
            Adjusted closing prices were used for return calculations because they
            include the effect of stock splits and other changes in the price history.
            """)

        st.markdown("#### Market Performance Indicators")

        with st.container(border=True):
            st.markdown("""
            The company metrics table already contained several transformed market
            indicators, including `cagr`, `cumulative_return`,
            `annualized_volatility`, and `max_drawdown`.

            These indicators were retained in the merged company dataset.

            Missing values in `last_close`, `cumulative_return`, `cagr`,
            `return_1y`, and `return_90d` were reconstructed from the daily
            stock-price table.

            Cumulative return was calculated as the last price divided by the first
            price, minus one.

            CAGR was calculated from the first price, the last price, and the length
            of the available price history.

            The one-year and 90-day returns were calculated from adjusted closing
            prices.

            Annualized volatility was used as the main company-level measure of
            stock-price variability, while maximum drawdown showed the largest fall
            from a previous price peak.
            """)

        st.divider()

        st.subheader("2. Annual Revenue Growth")

        with st.container(border=True):
            st.markdown("""
            For the analysis of revenue growth and stock returns, two additional
            annual variables were calculated.

            First, the financial observations were sorted by company and fiscal year.

            Annual revenue growth was then calculated separately for every company as:
            """)

            st.info(
                "Annual revenue growth = "
                "(Revenue in the current year / Revenue in the previous year) − 1"
            )

            st.markdown("""
            The calculation was performed inside each company group, so revenue
            values from different companies were not compared with each other.

            The first available financial year of every company had no previous
            observation and therefore received a missing growth value.
            """)
        st.subheader("3. Annual Stock Return")

        with st.container(border=True):
            st.markdown("""
            Annual stock return was calculated from the daily adjusted closing prices.

            For every company and calendar year, the first and last available
            adjusted closing prices were selected.

            The return was calculated as:
            """)

            st.info(
                "Annual stock return = "
                "(Last adjusted closing price of the year / "
                "First adjusted closing price of the year) − 1"
            )

            st.markdown("""
            The annual revenue growth table and the annual stock return table were
            then merged by ticker and year.

            Observations without both values were removed because a correlation could
            not be calculated for incomplete pairs.

            The final table used for this part of the analysis contained 121
            company-year observations from 43 companies.

            Several observations from the same company could be included because the
            calculations were performed for different years.
            """)

        st.divider()
        st.subheader("4. Macroeconomic Data Transformation")

        with st.container(border=True):
            st.markdown("""
            The macroeconomic dataset required a separate transformation because it
            was originally stored in long format.

            The name of each indicator was stored in `indicator_name`, while all
            numerical values were stored together in the `value` column.

            The table was transformed into wide format.

            After this transformation, every row represented one country-year pair,
            and every macroeconomic indicator was placed in a separate column.

            This made it possible to analyse GDP per capita, internet usage, account
            ownership, population, and other indicators separately.
            """)

        st.markdown("#### Missing Macroeconomic Values")

        with st.container(border=True):
            st.markdown("""
            Missing macroeconomic values were filled only within the same country.

            Forward fill was used when an earlier observation was available, while
            backward fill was used when the first available observation started after
            the beginning of the selected period.

            For Argentina, `gdp_total_usd` was missing, but GDP per capita and total
            population were available.

            Total GDP was therefore reconstructed as:
            """)

            st.info("Total GDP = GDP per capita × Population")

        st.divider()

        st.subheader("5. Country Clustering")

        with st.container(border=True):
            st.markdown("""
            Country clustering was then used to create a new country-group variable
            for the volatility hypothesis.

            Four indicators were selected: `gdp_per_capita_usd`,
            `internet_users_pct`, `account_ownership_pct_adult`, and
            `urban_population_pct`.

            Since these variables were measured on different scales, they were
            standardized before clustering.

            For every feature, its mean was subtracted and the result was divided by
            its standard deviation.

            After standardization, each indicator had a comparable scale and no
            variable was allowed to dominate the clustering only because its original
            numerical values were larger.
            """)

        st.markdown("#### K-Means Classification")

        with st.container(border=True):
            st.markdown("""
            The standardized country data was divided into two groups with K-Means
            clustering.

            The number of clusters was set to two.

            The cluster with the higher average GDP per capita was labelled
            `Developed`, while the other cluster was labelled `Emerging`.

            The resulting country classification was then merged with the company
            data through the country code.

            As a result, every company received a country-group label that could be
            compared with its annualized stock volatility.
            """)

        st.divider()

        st.subheader("6. Final Datasets for Hypothesis Testing")

        with st.container(border=True):
            st.markdown("""
            These transformations produced two final datasets for hypothesis testing.

            The first dataset contained annual revenue growth and annual stock return
            for matched company-year observations.

            The second dataset contained company-level annualized volatility together
            with the developed or emerging classification of the company’s country.

            All transformations were completed before the statistical tests were
            performed.
            """)

        st.success(
            "The transformed datasets are ready for the two hypothesis tests."
        )

    # H1
    with tab_h1:
        st.subheader("H1: Annual revenue growth is positively associated with annual stock returns")
        st.markdown("""
        **Hypothesis:** Company-years with higher annual revenue growth tend to have higher
        annual stock returns.

        **Method:** Spearman rank correlation is used because both variables are skewed and contain
        outliers. A cluster bootstrap resamples companies rather than individual rows, preserving the
        company-year panel structure.
        """)

        fin_growth = (
            df_fin
            .sort_values(["company_id", "fiscal_year"])
            .copy()
        )

        fin_growth["annual_revenue_growth"] = (
            fin_growth
            .groupby("company_id")["total_revenue"]
            .pct_change(fill_method=None)
        )

        fin_growth = (
            fin_growth[
                [
                    "company_id",
                    "ticker",
                    "fiscal_year",
                    "total_revenue",
                    "annual_revenue_growth"
                ]
            ]
            .rename(columns={"fiscal_year": "year"})
            .dropna(subset=["annual_revenue_growth"])
            .copy()
        )

        prices_daily = df_stick.copy()
        prices_daily["date"] = pd.to_datetime(prices_daily["date"], errors="coerce")
        prices_daily["adj_close"] = pd.to_numeric(prices_daily["adj_close"], errors="coerce")

        prices_daily = (
            prices_daily
            .dropna(subset=["ticker", "date", "adj_close"])
            .query("adj_close > 0")
            .drop_duplicates(subset=["ticker", "date"], keep="last")
            .sort_values(["ticker", "date"])
            .copy()
        )

        prices_daily["year"] = prices_daily["date"].dt.year

        annual_stock = (
            prices_daily
            .groupby(["ticker", "year"], as_index=False)
            .agg(
                first_date=("date", "first"),
                last_date=("date", "last"),
                first_price=("adj_close", "first"),
                last_price=("adj_close", "last"),
                trading_days=("adj_close", "count")
            )
        )

        annual_stock["annual_stock_return"] = (
            annual_stock["last_price"] / annual_stock["first_price"] - 1
        )

        h1_panel = (
            fin_growth
            .merge(
                annual_stock[
                    [
                        "ticker",
                        "year",
                        "first_price",
                        "trading_days",
                        "annual_stock_return"
                    ]
                ],
                on=["ticker", "year"],
                how="inner"
            )
            .dropna(subset=["annual_revenue_growth", "annual_stock_return"])
            .copy()
        )

        h1_panel = h1_panel[h1_panel["year"] <= 2025].copy()

        def company_bootstrap_spearman(data, cluster_col="ticker", n_boot=5000, seed=42):
            clean_data = (
                data[
                    [
                        cluster_col,
                        "annual_revenue_growth",
                        "annual_stock_return"
                    ]
                ]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            company_groups = [
                (
                    group["annual_revenue_growth"].to_numpy(),
                    group["annual_stock_return"].to_numpy()
                )
                for _, group in clean_data.groupby(cluster_col, sort=False)
            ]

            n_companies = len(company_groups)

            if n_companies < 2:
                return np.nan, np.nan, np.nan

            rng = np.random.default_rng(seed)
            bootstrap_corrs = np.empty(n_boot)

            for i in range(n_boot):
                sampled_indices = rng.integers(
                    low=0,
                    high=n_companies,
                    size=n_companies
                )

                x_sample = np.concatenate([
                    company_groups[index][0]
                    for index in sampled_indices
                ])

                y_sample = np.concatenate([
                    company_groups[index][1]
                    for index in sampled_indices
                ])

                bootstrap_corrs[i] = stats.spearmanr(
                    x_sample,
                    y_sample,
                    nan_policy="omit"
                ).statistic

            bootstrap_corrs = bootstrap_corrs[np.isfinite(bootstrap_corrs)]

            if len(bootstrap_corrs) == 0:
                return np.nan, np.nan, np.nan

            return np.percentile(bootstrap_corrs, [2.5, 50, 97.5])

        if len(h1_panel) > 5:
            spearman_result = stats.spearmanr(
                h1_panel["annual_revenue_growth"],
                h1_panel["annual_stock_return"],
                nan_policy="omit"
            )

            corr = spearman_result.statistic
            p_value = spearman_result.pvalue

            ci_low, bootstrap_corr, ci_high = company_bootstrap_spearman(
                h1_panel,
                cluster_col="ticker",
                n_boot=5000
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Company-year observations", f"{len(h1_panel):,}")
            col2.metric("Companies", f"{h1_panel['company_id'].nunique():,}")
            col3.metric("Spearman rho", f"{corr:.4f}")
            col4.metric("p-value", f"{p_value:.4f}")

            col5, col6, col7 = st.columns(3)
            col5.metric("Bootstrap median rho", f"{bootstrap_corr:.4f}")
            col6.metric("95% CI lower", f"{ci_low:.4f}")
            col7.metric("95% CI upper", f"{ci_high:.4f}")

            x_low, x_high = h1_panel["annual_revenue_growth"].quantile([0.02, 0.98])
            y_low, y_high = h1_panel["annual_stock_return"].quantile([0.02, 0.98])

            plot_data = h1_panel[
                h1_panel["annual_revenue_growth"].between(x_low, x_high)
                &
                h1_panel["annual_stock_return"].between(y_low, y_high)
            ].copy()

            plot_data["annual_revenue_growth_pct"] = (
                plot_data["annual_revenue_growth"] * 100
            )

            plot_data["annual_stock_return_pct"] = (
                plot_data["annual_stock_return"] * 100
            )

            fig, ax = plt.subplots(figsize=(9, 6))

            sns.regplot(
                data=plot_data,
                x="annual_revenue_growth_pct",
                y="annual_stock_return_pct",
                scatter_kws={
                    "alpha": 0.6,
                    "s": 45
                },
                line_kws={
                    "linewidth": 2,
                    "color": "black"
                },
                ax=ax
            )

            ax.axhline(0, linestyle="--", linewidth=1, color="gray")
            ax.axvline(0, linestyle="--", linewidth=1, color="gray")

            ax.set_title(
                "Annual Revenue Growth vs Annual Stock Return\n"
                f"Spearman rho={corr:.4f}, p={p_value:.4f}"
            )

            ax.set_xlabel("Annual Revenue Growth (%)")
            ax.set_ylabel("Annual Stock Return (%)")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(f"""
            **Result:** H1 is **supported**. Spearman's rank correlation is positive
            (rho={corr:.4f}) and statistically significant (p={p_value:.4f}). The company-cluster
            bootstrap also gives a positive median rho={bootstrap_corr:.4f}, with a 95% confidence
            interval [{ci_low:.4f}, {ci_high:.4f}], which stays above zero.
            """)
            st.markdown("""
The first hypothesis tested whether annual revenue growth was positively associated with annual stock return. The analysis was based on 121 company-year observations from 43 companies. Since the distributions contained extreme values and the relationship was not expected to be perfectly linear, Spearman’s rank correlation was used instead of Pearson’s correlation. The estimated Spearman correlation coefficient was 0.2267. This value shows a positive but relatively weak relationship: company-year observations with higher annual revenue growth generally tended to have higher annual stock returns, but the relationship was not strong and many observations did not follow the same pattern. This can also be seen in the scatter plot, where the fitted line has a positive slope, while the individual points remain widely distributed around it. The p-value was 0.0124, which is below the standard significance level of 0.05. Therefore, the null hypothesis of no relationship was rejected.

Because several yearly observations belonged to the same companies, an additional cluster bootstrap was performed at the company level. Entire companies, rather than individual company-year rows, were resampled during each bootstrap iteration. The bootstrap median correlation was 0.2229, which is very close to the original Spearman coefficient of 0.2267. The 95% bootstrap confidence interval ranged from 0.0763 to 0.3732. Since the full interval is above zero, the positive relationship remained stable after the dependence between observations from the same company was considered. Therefore, the hypothesis was supported by the data. However, the size of the correlation remained small, so annual revenue growth alone cannot explain most of the differences in annual stock returns. The result shows a statistically significant positive association, but not a strong or deterministic relationship.

""")
        else:
            st.warning("Not enough company-year observations to run H1.")

    # H2
    with tab_h2:
        st.subheader("H2: Emerging-market countries show higher post-IPO volatility")
        st.markdown("""
        **Hypothesis:** Companies from emerging-market countries exhibit higher annualized rolling
        volatility after IPO than companies from developed-market countries.

        **Method:** Countries are classified with K-Means using macroeconomic features. Daily
        log returns are converted to 252-trading-day annualized rolling volatility, then aggregated
        first to companies and then to countries. A permutation test compares country-level means.
        """)

        from matplotlib.ticker import PercentFormatter

        company_country = (
            df_filled[
                ["company_id", "country_code_x"]
            ]
            .rename(columns={"country_code_x": "country_code"})
            .dropna()
            .drop_duplicates()
        )

        duplicate_companies = company_country["company_id"].duplicated(keep=False)

        if duplicate_companies.any():
            problem_companies = (
                company_country.loc[
                    duplicate_companies,
                    "company_id"
                ]
                .unique()
            )

            st.error(f"Several countries found for companies: {problem_companies}")
        else:
            stock_vol = df_stick.copy()

            stock_vol["date"] = pd.to_datetime(stock_vol["date"], errors="coerce")
            stock_vol["adj_close"] = pd.to_numeric(stock_vol["adj_close"], errors="coerce")

            stock_vol = (
                stock_vol
                .dropna(subset=["company_id", "date", "adj_close"])
                .loc[lambda data: data["adj_close"] > 0]
                .sort_values(["company_id", "date"])
                .copy()
            )

            stock_vol = stock_vol.merge(
                company_country,
                on="company_id",
                how="left",
                validate="many_to_one"
            )

            stock_vol = stock_vol.merge(
                country_df[
                    ["country_code", "market_type"]
                ],
                on="country_code",
                how="left",
                validate="many_to_one"
            )

            stock_vol["log_return"] = (
                stock_vol
                .groupby("company_id")["adj_close"]
                .transform(lambda series: np.log(series).diff())
            )

            stock_vol["rolling_vol"] = (
                stock_vol
                .groupby("company_id")["log_return"]
                .transform(
                    lambda series: series.rolling(
                        window=252,
                        min_periods=126
                    ).std() * np.sqrt(252)
                )
            )

            stock_vol["year"] = stock_vol["date"].dt.year

            company_year_vol = (
                stock_vol
                .groupby(
                    [
                        "company_id",
                        "country_code",
                        "market_type",
                        "year"
                    ]
                )["rolling_vol"]
                .median()
                .reset_index()
                .dropna()
            )

            vol_by_year = (
                company_year_vol
                .groupby(["year", "market_type"])["rolling_vol"]
                .median()
                .reset_index()
            )

            vol_per_company = (
                stock_vol
                .groupby(
                    [
                        "company_id",
                        "country_code",
                        "market_type"
                    ]
                )["rolling_vol"]
                .median()
                .reset_index()
                .dropna()
            )

            vol_per_country = (
                vol_per_company
                .groupby(
                    ["country_code", "market_type"]
                )["rolling_vol"]
                .median()
                .reset_index()
            )

            developed = (
                vol_per_country.loc[
                    vol_per_country["market_type"] == "Developed",
                    "rolling_vol"
                ]
                .to_numpy()
            )

            emerging = (
                vol_per_country.loc[
                    vol_per_country["market_type"] == "Emerging",
                    "rolling_vol"
                ]
                .to_numpy()
            )

            if len(developed) < 2 or len(emerging) < 2:
                st.warning("Not enough country-level observations to run H2.")
            else:
                observed_diff = emerging.mean() - developed.mean()

                combined = np.concatenate([emerging, developed])
                n_emerging = len(emerging)

                n_permutations = 20000
                perm_diffs = np.empty(n_permutations)

                rng = np.random.default_rng(42)

                for i in range(n_permutations):
                    shuffled = rng.permutation(combined)

                    perm_emerging = shuffled[:n_emerging]
                    perm_developed = shuffled[n_emerging:]

                    perm_diffs[i] = (
                        perm_emerging.mean()
                        - perm_developed.mean()
                    )

                p_value_two_sided = (
                    np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) + 1
                ) / (n_permutations + 1)

                p_value_greater = (
                    np.sum(perm_diffs >= observed_diff) + 1
                ) / (n_permutations + 1)

                d = cohen_d(pd.Series(emerging), pd.Series(developed))

                abs_d = abs(d)

                if abs_d < 0.2:
                    effect = "negligible"
                elif abs_d < 0.5:
                    effect = "small"
                elif abs_d < 0.8:
                    effect = "medium"
                else:
                    effect = "large"

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Developed countries", f"{len(developed):,}")
                col2.metric("Emerging countries", f"{len(emerging):,}")
                col3.metric("Emerging mean vol", f"{emerging.mean():.2%}")
                col4.metric("Developed mean vol", f"{developed.mean():.2%}")

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Observed difference", f"{observed_diff:.2%}")
                col6.metric("Two-sided p-value", f"{p_value_two_sided:.4f}")
                col7.metric("One-sided p-value", f"{p_value_greater:.4f}")
                col8.metric(f"Cohen's d ({effect})", f"{d:.3f}")

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                order = ["Developed", "Emerging"]

                sns.boxplot(
                    data=vol_per_country,
                    x="market_type",
                    y="rolling_vol",
                    order=order,
                    hue="market_type",
                    palette={
                        "Developed": "steelblue",
                        "Emerging": "coral"
                    },
                    legend=False,
                    width=0.5,
                    ax=axes[0]
                )

                sns.stripplot(
                    data=vol_per_country,
                    x="market_type",
                    y="rolling_vol",
                    order=order,
                    color="black",
                    size=7,
                    jitter=0.08,
                    ax=axes[0]
                )

                axes[0].set_title(
                    "Median Rolling Volatility by Country\n"
                    f"two-sided p={p_value_two_sided:.4f} | "
                    f"d={d:.3f}"
                )

                axes[0].set_xlabel("Market cluster")
                axes[0].set_ylabel("Median Annualized Rolling Volatility")
                axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))

                for market_type, color in [
                    ("Developed", "steelblue"),
                    ("Emerging", "coral")
                ]:
                    data = vol_by_year[
                        vol_by_year["market_type"] == market_type
                    ]

                    axes[1].plot(
                        data["year"],
                        data["rolling_vol"],
                        color=color,
                        linewidth=2,
                        marker="o",
                        label=market_type
                    )

                axes[1].axvline(
                    2020,
                    color="gray",
                    linestyle="--",
                    alpha=0.6,
                    label="2020"
                )

                axes[1].axvline(
                    2022,
                    color="orange",
                    linestyle="--",
                    alpha=0.6,
                    label="2022"
                )

                axes[1].set_title("Median Annualized Rolling Volatility Over Time")
                axes[1].set_xlabel("Year")
                axes[1].set_ylabel("Median Annualized Rolling Volatility")
                axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
                axes[1].legend()

                axes[0].text(
                    0.98,
                    0.95,
                    f"Effect: {effect}",
                    transform=axes[0].transAxes,
                    fontsize=9,
                    ha="right",
                    va="top",
                    color="gray"
                )

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(
                    vol_per_country
                    .sort_values(["market_type", "rolling_vol"])
                    .assign(
                        rolling_vol=lambda data: data["rolling_vol"].map("{:.2%}".format)
                    ),
                    use_container_width=True
                )

                st.markdown(f"""
                **Result:** H2 is **not confirmed**. Emerging-market countries have a slightly higher
                mean volatility ({emerging.mean():.2%} vs {developed.mean():.2%}), but the difference is
                not statistically significant: two-sided p={p_value_two_sided:.4f}, one-sided
                p={p_value_greater:.4f}. Cohen's d={d:.3f}, which corresponds to a {effect} effect.
                """)
                st.markdown("""
A permutation test was used to evaluate whether the observed difference could have appeared by chance. The two-sided p-value was 0.4193, which is much higher than the standard significance level of 0.05. The one-sided p-value for the specific hypothesis that emerging-market volatility was higher was 0.2058, which is also above 0.05. Therefore, the null hypothesis could not be rejected.
The observed difference of 4.62 percentage points was not statistically significant. Cohen’s d was equal to 0.459, which indicates a small effect according to the classification used in the analysis. This means that the emerging group had somewhat higher volatility in the observed sample, but the difference was not large enough or stable enough to support a general conclusion. 
The time-series plot also shows that volatility changed considerably over time in both groups. Developed markets had higher median volatility in most years, especially around 2020–2023, while emerging markets were slightly higher in some individual years, such as 2019 and 2025. Both groups reached elevated volatility around 2020–2022 and then declined. Since the lines cross and the size of the difference changes from year to year, no stable separation between the groups can be observed. 
Overall, the hypothesis was not supported by the statistical test. Emerging countries had a higher average volatility in this sample, but the evidence was insufficient to conclude that emerging-market e-commerce companies are systematically more volatile than developed-market companies.
              
""")

# ══════════════════════════════════════════
# 8. DISCUSSION
# ══════════════════════════════════════════

elif page == "Discussion":
    st.title("Discussion")
    st.markdown("""
    ### Summary of Findings

    **Data quality.** The dataset required non-trivial imputation work. Return metrics for 13 Indian 
    companies were reconstructed from raw price data using verified formulas. GoTo Group's 
    operating cash flow was recovered algebraically. Country classifications were derived 
    via unsupervised K-Means clustering rather than arbitrary cutoffs.

    **H1 — Macro environment does not predict CAGR.**  
    The OLS regression (R²=0.076, F p=0.664) found no statistically significant relationship 
    between internet penetration, GDP per capita, or FDI inflows and post-IPO CAGR. 
    A likely explanation is that firm-level factors (business model, timing of listing, 
    competitive position) dominate country-level macro conditions for this asset class. 
    The small sample size (n=43) also limits statistical power.

    **H2 — Emerging markets are not more volatile.**  
    Contrary to expectations, developed-market companies show slightly *higher* median rolling 
    volatility (Cohen's d=−0.41, medium effect, p=0.19). North American e-commerce companies — 
    many of which are high-growth, loss-making platforms — exhibit wide return distributions. 
    Indian companies in contrast show tighter volatility profiles, possibly reflecting 
    more established domestic businesses at the time of listing.

    **Limitations.**
    - Sample size (43 companies) restricts statistical power for detecting moderate effects.
    - Financial data covers only 2022–2025; long-run fundamentals are unavailable.
    - Country clustering on 4 macro features is a simplification; richer classifications exist.
    - Survivorship bias: only publicly traded companies are included.

    ### Directions for further research
    - Extend the financial panel backwards using alternative data sources.
    - Include underpricing (first-day return) as an additional performance metric.
    - Apply panel regression with fixed effects to control for unobserved firm heterogeneity.
    - Test sector-level hypotheses with a larger sample from a single exchange.
    """)
