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
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="E-commerce IPO Analysis", layout="wide", page_icon="📈")

# ──────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────

@st.cache_data
def load_data():
    df_comp     = pd.read_csv("data/companies.csv")
    df_metrics  = pd.read_csv("data/company_metrics.csv")
    df_ecommerce= pd.read_csv("data/ecommerce_index.csv")
    df_fin      = pd.read_csv("data/financials_annual.csv")
    df_macro    = pd.read_csv("data/macro_indicators.csv")
    df_stick    = pd.read_csv("data/prices_daily.csv")
    return df_comp, df_metrics, df_ecommerce, df_fin, df_macro, df_stick

@st.cache_data
def prepare_data(df_comp, df_metrics, df_fin, df_macro, df_stick):
    # merge companies
    df = df_comp.merge(df_metrics, on="company_id", how="inner")

    # drop columns
    cols_to_drop = ["headquarters","founders","employees","employees_year",
                    "name","reporting_currency","wikipedia_title","wikidata_qid",
                    "wiki_title_canonical","website","wiki_extract","wiki_url","wiki_thumbnail"]
    df_filled = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # fix founded_date
    fixes = {"carvana":"2012-01-04","sea":"2009-05-08",
             "delhivery":"2011-05-01","affirm":"2012-01-17"}
    for cid, date in fixes.items():
        df_filled.loc[df_filled["company_id"]==cid, "founded_date"] = date
    df_filled["founded_date"] = (
        df_filled["founded_date"]
        .str.replace("-00-","-01-",regex=False)
        .str.replace(r"-00$","-01",regex=True)
    )
    df_filled["founded_date"] = pd.to_datetime(df_filled["founded_date"], errors="coerce")

    # fill missing return values from stock prices
    df_stick2 = df_stick.copy()
    df_stick2["date"] = pd.to_datetime(df_stick2["date"])
    for idx in df_filled.index:
        if pd.notna(df_filled.loc[idx,"cumulative_return"]):
            continue
        cid = df_filled.loc[idx,"company_id"]
        tmp = df_stick2[df_stick2["company_id"]==cid].sort_values("date").reset_index(drop=True)
        if len(tmp) < 2:
            continue
        first_price = tmp.iloc[0]["close"]
        last_price  = tmp.iloc[-2]["close"]
        last_date   = tmp.iloc[-2]["date"]
        df_filled.loc[idx,"last_close"] = last_price
        df_filled.loc[idx,"cumulative_return"] = (last_price/first_price)-1
        hs = pd.to_datetime(df_filled.loc[idx,"history_start"])
        he = pd.to_datetime(df_filled.loc[idx,"history_end"])
        years = (he-hs).days/365.25
        if years>0:
            df_filled.loc[idx,"cagr"] = ((last_price/first_price)**(1/years))-1
        adj_last = tmp.iloc[-2]["adj_close"]
        adj_first= tmp.iloc[0]["adj_close"]
        t1y = last_date - pd.DateOffset(years=1)
        s1y = tmp.loc[tmp["date"]<=t1y,"close"]
        if len(s1y)>0:
            df_filled.loc[idx,"return_1y"] = (adj_last/s1y.iloc[-1])-1
        t90 = last_date - pd.DateOffset(days=128)
        s90 = tmp.loc[tmp["date"]<=t90,"adj_close"]
        if len(s90)>0:
            df_filled.loc[idx,"return_90d"] = (adj_last/s90.iloc[-1])-1

    # financials cleanup
    df_fin2 = df_fin.copy()
    critical_cols = ["net_income","operating_income","total_revenue","ebitda",
                     "total_assets","total_liabilities","cash_and_equivalents",
                     "total_equity","capex","free_cash_flow"]
    mask = df_fin2[critical_cols].isna().all(axis=1)
    df_fin2 = df_fin2.loc[~mask].copy()
    df_fin2["operating_margin"] = df_fin2["operating_income"]/df_fin2["total_revenue"]
    df_fin2["rd_intensity"] = df_fin2["research_dev"]/df_fin2["total_revenue"]
    df_fin2["has_rd"] = df_fin2["research_dev"].notna().astype(int)
    goto_mask = df_fin2["ticker"]=="GOTO.JK"
    df_fin2.loc[goto_mask,"operating_cash_flow"] = (
        df_fin2.loc[goto_mask,"free_cash_flow"] - df_fin2.loc[goto_mask,"capex"]
    )

    # macro
    macro_wide = (
        df_macro.pivot_table(index=["country_code","year"],
                             columns="indicator_name", values="value")
        .reset_index()
        .sort_values(["country_code","year"])
    )
    macro_wide["account_ownership_pct_adult"] = (
        macro_wide.groupby("country_code")["account_ownership_pct_adult"].ffill().bfill()
    )
    m_mask = macro_wide["gdp_total_usd"].isna()
    macro_wide.loc[m_mask,"gdp_total_usd"] = (
        macro_wide.loc[m_mask,"gdp_per_capita_usd"]*macro_wide.loc[m_mask,"population_total"]
    )
    for col in ["gdp_per_capita_usd","internet_users_pct","fdi_inflow_pct_gdp",
                "mobile_subs_per_100","gdp_total_usd","population_total","urban_population_pct"]:
        macro_wide[col] = macro_wide.groupby("country_code")[col].ffill().bfill()

    # country clustering
    features = ["gdp_per_capita_usd","internet_users_pct",
                "account_ownership_pct_adult","urban_population_pct"]
    country_df = macro_wide.groupby("country_code")[features].mean().reset_index()
    X = country_df[features].values
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    centroids, _ = kmeans(X_scaled, 2, seed=42)
    labels, _ = vq(X_scaled, centroids)
    country_df["cluster"] = labels
    dev_cluster = country_df.groupby("cluster")["gdp_per_capita_usd"].mean().idxmax()
    country_df["market_type"] = country_df["cluster"].apply(
        lambda x: "Developed" if x==dev_cluster else "Emerging"
    )
    X_centered = X_scaled - X_scaled.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    coords = X_centered @ Vt.T[:,:2]
    country_df["pca1"] = coords[:,0]
    country_df["pca2"] = coords[:,1]

    # df_companies
    macro_latest = (
        macro_wide.sort_values("year")
        .groupby("country_code").last().reset_index()
    )[["country_code","internet_users_pct","gdp_per_capita_usd","fdi_inflow_pct_gdp"]]

    df_fin_latest = (
        df_fin2.sort_values("fiscal_year")
        .groupby("company_id").last().reset_index()
    )[["company_id","operating_margin","free_cash_flow","total_revenue",
       "total_assets","ebitda","rd_intensity","has_rd","operating_cash_flow"]]

    df_companies = df_filled.drop_duplicates("company_id").copy()
    df_companies = df_companies.merge(df_fin_latest, on="company_id", how="left")
    df_companies = df_companies.merge(macro_latest, left_on="country_code_x",
                                      right_on="country_code", how="left")
    if "country_code" in df_companies.columns:
        df_companies = df_companies.drop(columns=["country_code"])
    df_companies = df_companies.merge(
        country_df[["country_code","market_type","cluster"]],
        left_on="country_code_x", right_on="country_code", how="left"
    )

    # stock copy with rolling vol
    df_stock_copy = df_stick.copy()
    df_stock_copy = df_stock_copy.merge(
        df_filled[["company_id","country_code_x"]].drop_duplicates(),
        on="company_id", how="left"
    ).rename(columns={"country_code_x":"country_code"})
    df_stock_copy = df_stock_copy.merge(
        country_df[["country_code","market_type"]], on="country_code", how="left"
    )
    df_stock_copy = df_stock_copy.sort_values(["company_id","date"])
    df_stock_copy["daily_return"] = (
        df_stock_copy.groupby("company_id")["adj_close"].pct_change()
    )
    df_stock_copy["rolling_vol"] = (
        df_stock_copy.groupby("company_id")["daily_return"]
        .transform(lambda x: x.rolling(252).std()*np.sqrt(252))
    )
    df_stock_copy["year"] = pd.to_datetime(df_stock_copy["date"]).dt.year

    return df_filled, df_fin2, macro_wide, country_df, df_companies, df_stock_copy


def cohen_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.var(ddof=1)+(n2-1)*g2.var(ddof=1))/(n1+n2-2))
    return (g1.mean()-g2.mean())/pooled


# ──────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────

df_comp, df_metrics, df_ecommerce, df_fin_raw, df_macro, df_stick = load_data()
df_filled, df_fin, macro_wide, country_df, df_companies, df_stock_copy = prepare_data(
    df_comp, df_metrics, df_fin_raw, df_macro, df_stick
)

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
    regions and macroeconomic environments. Using a composite dataset of 43 publicly traded e-commerce 
    firms spanning multiple exchanges (NYSE, NASDAQ, LSE, NSE, TSE and others), we combine company-level 
    financial metrics, daily stock prices, and country-level macroeconomic indicators to test two central 
    hypotheses.

    **H1** examines whether macroeconomic conditions — specifically internet penetration, GDP per capita, 
    and FDI inflows — moderate long-run stock performance (CAGR) after IPO.  
    **H2** investigates whether companies from emerging markets exhibit systematically higher post-IPO 
    volatility than their developed-market counterparts.

    The analysis pipeline includes missing-value imputation from raw price data, formula verification 
    against reported metrics, K-Means country clustering into Developed/Emerging groups, bootstrap 
    correlation matrices, permutation tests, Cohen's d effect-size estimation, and OLS regression 
    with heteroscedasticity-robust standard errors.

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
    The project uses **5 interconnected datasets** collected from public sources:

    | Dataset | Rows | Key fields |
    |---|---|---|
    | `companies.csv` | 43 | company_id, exchange, country, founded_date, industry |
    | `company_metrics.csv` | 43 | CAGR, volatility, cumulative_return, max_drawdown |
    | `financials_annual.csv` | ~172 | revenue, net_income, EBITDA, free_cash_flow (2022–2025) |
    | `macro_indicators.csv` | ~1500 | GDP, internet_users_pct, FDI by country & year (2011–2024) |
    | `prices_daily.csv` | ~85K | daily OHLCV prices per company |
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Companies by Region")
        st.bar_chart(df_filled["region_x"].value_counts())
    with col2:
        st.subheader("Companies by Segment")
        st.bar_chart(df_filled["segment_x"].value_counts())

    st.subheader("Missing Values in Main Dataset")
    missing = df_filled.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    st.dataframe(missing.rename("missing_count").to_frame())

    st.subheader("Sample rows")
    st.dataframe(df_filled.head(5))

# ══════════════════════════════════════════
# 3. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════

elif page == "Descriptive Statistics":
    st.title("Descriptive Statistics")

    st.markdown("""
    We group numeric fields into three conceptual buckets and report percentile-aware statistics.
    The large gap between **mean** and **median** in market metrics signals right-skewed distributions 
    driven by a handful of extreme performers.
    """)

    market_cols = ["cagr","annualized_volatility","cumulative_return","max_drawdown","return_90d","return_1y"]
    st.subheader("Market Performance Metrics")
    st.dataframe(df_filled[market_cols].describe(percentiles=[.05,.25,.5,.75,.95]).round(4))

    fin_cols = ["total_revenue","net_income","operating_income",
                "total_assets","free_cash_flow","net_margin","operating_margin","revenue_cagr"]
    fin_available = [c for c in fin_cols if c in df_filled.columns]
    st.subheader("Financial Metrics (latest fiscal year, from companies dataset)")
    st.dataframe(df_filled[fin_available].describe(percentiles=[.05,.25,.5,.75,.95]).round(4))

    macro_cols = ["gdp_per_capita_usd","internet_users_pct","fdi_inflow_pct_gdp"]
    macro_available = [c for c in macro_cols if c in df_companies.columns]
    st.subheader("Macroeconomic Indicators (latest year per country)")
    st.dataframe(df_companies[macro_available].describe(percentiles=[.05,.25,.5,.75,.95]).round(4))

    st.subheader("Top 5 most skewed numeric columns")
    num_cols = df_filled.select_dtypes(include=np.number).columns
    skew = df_filled[num_cols].skew().abs().sort_values(ascending=False).head(5)
    st.dataframe(skew.rename("abs_skewness").to_frame())

# ══════════════════════════════════════════
# 4. DATA CLEANUP
# ══════════════════════════════════════════

elif page == "Data Cleanup":
    st.title("Data Cleanup")

    st.markdown("""
    ### Steps performed

    **1. Column removal** — 13 columns with low informational value or high missingness 
    (headquarters, founders, employees, wiki fields) were dropped.

    **2. Formula verification** — All return metrics (cumulative_return, CAGR, return_1y, return_90d) 
    were reverse-engineered from raw price data and compared against reported values. 
    Maximum deviation was < 1e-8, confirming formula correctness.

    **3. Missing return imputation** — 13 Indian companies had missing return metrics. 
    Values were reconstructed from `prices_daily.csv` using the verified formulas.

    **4. Founded date fixes** — 4 companies had missing or malformed dates (e.g. `2012-00-00`). 
    Corrected manually via external lookup.

    **5. Financials cleanup** — Rows where all critical financial columns were NaN (first reporting year) 
    were dropped. `operating_cash_flow` for GoTo Group (GOTO.JK) was reconstructed as `FCF − capex`, 
    verified on all other companies (0% deviation).

    **6. Macro imputation** — `account_ownership_pct_adult` is only surveyed every 3–4 years; 
    gaps filled with forward-fill then backward-fill within each country. `gdp_total_usd` 
    for Argentina reconstructed as `gdp_per_capita × population`.

    **7. Country clustering** — K-Means (k=2) on standardised macro features 
    (GDP/capita, internet %, account ownership, urban %) assigned each country to 
    Developed or Emerging market cluster.
    """)

    st.subheader("Missing values after cleanup")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Companies dataset**")
        miss = df_filled.isna().sum()
        miss = miss[miss > 0]
        st.dataframe(miss.rename("remaining_NaN"))
    with col2:
        st.write("**Financials dataset**")
        miss_fin = df_fin.isna().sum()
        miss_fin = miss_fin[miss_fin > 0]
        st.dataframe(miss_fin.rename("remaining_NaN"))

    st.subheader("Country clustering result (PCA projection)")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Developed": "#2196F3", "Emerging": "#FF5722"}
    for mtype, group in country_df.groupby("market_type"):
        ax.scatter(group["pca1"], group["pca2"],
                   label=mtype, color=colors[mtype], s=100, alpha=0.8)
        for _, row in group.iterrows():
            ax.annotate(row["country_code"], (row["pca1"], row["pca2"]),
                        fontsize=8, alpha=0.7)
    ax.set_title("Country Clusters: Developed vs Emerging")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════
# 5. BASIC PLOTS
# ══════════════════════════════════════════

elif page == "Basic Plots":
    st.title("Basic Plots")

    # ── CAGR Histogram ──
    st.subheader("1. CAGR Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df_filled["cagr"].dropna(), bins=25, kde=True, ax=ax, color="steelblue")
    ax.axvline(df_filled["cagr"].median(), color="red", linestyle="--",
               label=f"Median = {df_filled['cagr'].median():.3f}")
    ax.axvline(df_filled["cagr"].mean(), color="orange", linestyle=":",
               label=f"Mean = {df_filled['cagr'].mean():.3f}")
    ax.set_title("CAGR Distribution (43 E-commerce Companies)")
    ax.legend()
    st.pyplot(fig)
    plt.close()
    st.markdown("""
    The distribution is approximately flat with a slight right skew. Most companies have CAGR between 
    −0.2 and +0.3. The mean and median are close to zero, suggesting that on average e-commerce IPOs 
    have not delivered sustained positive returns.
    """)

    # ── Net Margin vs Cumulative Return ──
    st.subheader("2. Net Margin vs Cumulative Return")
    df_plot = df_filled[df_filled["cumulative_return"]>0].copy()
    if "net_margin" in df_plot.columns:
        df_plot["net_margin_c"] = df_plot["net_margin"].clip(-0.15, 0.35)
        df_plot["cum_log"] = np.log(df_plot["cumulative_return"])

        fig = plt.figure(figsize=(12, 7))
        gs_obj = gs_module.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[1,4],
                                    hspace=0.05, wspace=0.05)
        ax_main  = fig.add_subplot(gs_obj[1,0])
        ax_top   = fig.add_subplot(gs_obj[0,0], sharex=ax_main)
        ax_right = fig.add_subplot(gs_obj[1,1], sharey=ax_main)

        regions = df_plot["region_x"].astype("category")
        ax_main.scatter(df_plot["net_margin_c"], df_plot["cum_log"],
                        c=regions.cat.codes, cmap="tab10", alpha=0.7, s=60)
        x = df_plot["net_margin_c"]
        y = df_plot["cum_log"]
        b, m_val = np.polyfit(x, y, 1)
        ax_main.plot(sorted(x), [b+m_val*xi for xi in sorted(x)],
                     color="red", linewidth=1.5, linestyle="--")
        ax_main.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax_main.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax_main.set_xlabel("Net Margin")
        ax_main.set_ylabel("Cumulative Return (log)")

        ax_top.hist(df_plot["net_margin_c"], bins=20, color="steelblue", alpha=0.7)
        ax_top.axis("off")
        ax_right.hist(df_plot["cum_log"], bins=20, orientation="horizontal",
                      color="steelblue", alpha=0.7)
        ax_right.axis("off")

        handles = [plt.scatter([],[],c=f"C{i}",label=r,s=40)
                   for i,r in enumerate(regions.cat.categories)]
        ax_main.legend(handles=handles, fontsize=7, loc="upper left")
        fig.suptitle("Net Margin vs Cumulative Return by Region", fontsize=13)
        st.pyplot(fig)
        plt.close()
        st.markdown("Marginal histograms show the distribution of each axis. "
                    "A weak positive trend is visible but driven by a few outliers.")

    # ── Violin ──
    st.subheader("3. Volatility Distribution by Region")
    df_vp = df_filled[df_filled["annualized_volatility"] <
                      df_filled["annualized_volatility"].quantile(0.95)]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(data=df_vp, x="region_x", y="annualized_volatility",
                   inner="box", ax=ax, color="steelblue", alpha=0.7)
    means = df_vp.groupby("region_x")["annualized_volatility"].mean()
    for i, tick in enumerate(ax.get_xticklabels()):
        r = tick.get_text()
        if r in means.index:
            ax.scatter(i, means[r], color="red", zorder=5, s=60, marker="D",
                       label="Mean" if i==0 else "")
    ax.legend(fontsize=8)
    ax.set_title("Annualized Volatility by Region (◆ = mean)")
    plt.xticks(rotation=20)
    st.pyplot(fig)
    plt.close()
    st.markdown("All regions cluster around 0.5–0.6 median volatility. "
                "India shows the narrowest distribution; North America the widest upper tail.")

    # ── E-commerce index + macro ──
    st.subheader("4. E-commerce Growth vs Internet Penetration")
    ecom = df_ecommerce.copy()
    ecom["date"] = pd.to_datetime(ecom["date"])
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    ax1.plot(ecom["date"], ecom["index_level"], color="steelblue",
             linewidth=2, label="E-commerce Index")
    for country in macro_wide["country_code"].unique():
        data = macro_wide[macro_wide["country_code"]==country].copy()
        dates = pd.to_datetime(data["year"].astype(int).astype(str)+"-06-01")
        ax2.plot(dates.values, data["internet_users_pct"].values,
                 alpha=0.2, linewidth=1, color="coral")
    for country, color in [("US","red"),("CN","gold"),("IN","green"),("GB","purple")]:
        data = macro_wide[macro_wide["country_code"]==country]
        if len(data)==0: continue
        dates = pd.to_datetime(data["year"].astype(int).astype(str)+"-06-01")
        ax2.plot(dates.values, data["internet_users_pct"].values,
                 linewidth=2, color=color, label=f"Internet % ({country})")
    ax1.set_ylabel("E-commerce Index", color="steelblue")
    ax2.set_ylabel("Internet Users %", color="coral")
    ax2.set_ylim(0,100)
    lines1,labels1 = ax1.get_legend_handles_labels()
    lines2,labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper left")
    ax1.set_title("E-commerce Index vs Internet Penetration by Country")
    st.pyplot(fig)
    plt.close()
    st.markdown("Both the e-commerce index and internet penetration rise steadily until the COVID peak "
                "(2020–2021), after which the index corrects while internet adoption continues to grow.")

# ══════════════════════════════════════════
# 6. DETAILED OVERVIEW
# ══════════════════════════════════════════

elif page == "Detailed Overview":
    st.title("Detailed Overview")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Cumulative Return KDE",
        "CAGR by Region",
        "Equal-Weighted Portfolio",
        "Bootstrap Correlation"
    ])

    with tab1:
        st.subheader("Distribution of Cumulative Returns")
        fig, ax = plt.subplots(figsize=(10, 5))
        returns_col = df_filled["cumulative_return"].dropna()
        returns_col.plot.kde(ax=ax, linewidth=2.5)
        ax.axvline(returns_col.median(), linestyle="--", linewidth=2,
                   label=f"Median = {returns_col.median():.2f}")
        ax.axvline(returns_col.mean(), linestyle=":", linewidth=2,
                   label=f"Mean = {returns_col.mean():.2f}")
        for x in returns_col:
            ax.axvline(x, ymin=0, ymax=0.02, alpha=0.3, linewidth=1, color="gray")
        ax.set_title(f"Distribution of Cumulative Returns ({len(returns_col)} Companies)")
        ax.set_xlabel("Cumulative Return")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        plt.close()
        st.markdown("The distribution is strongly right-skewed. The median of 0.55 vs mean of 4.60 "
                    "indicates a small number of extreme winners distort the average.")

    with tab2:
        st.subheader("CAGR Distribution by Region (GridSpec)")

        # use yearly data from stock prices
        prices = df_stick.copy()
        prices["date"] = pd.to_datetime(prices["date"])
        prices["year"] = prices["date"].dt.year
        yearly = (prices.sort_values(["ticker","date"])
                  .groupby(["ticker","year"]).last().reset_index())
        first_price = yearly.groupby("ticker")["adj_close"].first()
        first_year  = yearly.groupby("ticker")["year"].first()
        yearly["first_price"] = yearly["ticker"].map(first_price)
        yearly["first_year"]  = yearly["ticker"].map(first_year)
        yearly["years_elapsed"] = yearly["year"] - yearly["first_year"]
        yearly = yearly[yearly["years_elapsed"]>0]
        yearly["cagr"] = ((yearly["adj_close"]/yearly["first_price"])**(1/yearly["years_elapsed"]))-1
        region_map = (df_filled[["ticker_x","region_x"]].drop_duplicates()
                      .rename(columns={"ticker_x":"ticker"}))
        yearly = yearly.merge(region_map, on="ticker", how="left")

        regions = df_filled["region_x"].dropna().unique()
        n = len(regions)
        overall_median = yearly["cagr"].median()
        fig = plt.figure(figsize=(14, 3*n))
        gs_obj = gs_module.GridSpec(n, 1, hspace=0.6)
        for i, region in enumerate(regions):
            ax = fig.add_subplot(gs_obj[i])
            data = yearly[yearly["region_x"]==region]["cagr"].dropna()
            if len(data)<3:
                ax.set_title(f"{region} (n={len(data)}) — too few observations")
                continue
            low, high = data.quantile(0.05), data.quantile(0.95)
            sns.kdeplot(data.clip(low,high), ax=ax, fill=True, alpha=0.3, linewidth=2)
            ax.axvline(data.median(), color="steelblue", linestyle="--", linewidth=1.5,
                       label=f"Median = {data.median():.2f}")
            ax.axvline(overall_median, color="red", linestyle=":", linewidth=1, alpha=0.7,
                       label=f"Overall = {overall_median:.2f}")
            ax.axvline(0, color="black", linewidth=0.8, alpha=0.3)
            ax.set_title(f"{region}  (n={len(data)})", fontsize=11, fontweight="bold")
            ax.set_xlabel("CAGR" if i==n-1 else "")
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_xlim(-1,1)
        fig.suptitle("CAGR Distribution by Region", fontsize=14, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Equal-Weighted Portfolio vs E-commerce Index")
        START_YEAR = 2015
        FORMATION_DATE = pd.Timestamp("2015-12-31")
        prices2 = df_stick.copy()
        prices2["date"] = pd.to_datetime(prices2["date"])
        first_dates = prices2.groupby("ticker")["date"].min()
        eligible = first_dates[first_dates<=FORMATION_DATE].index
        prices2 = prices2[prices2["ticker"].isin(eligible)].copy()
        prices2 = prices2[prices2["date"].dt.year >= START_YEAR]
        prices2 = prices2.sort_values(["ticker","date"])
        prices2["normalized"] = (prices2["adj_close"] /
                                  prices2.groupby("ticker")["adj_close"].transform("first") * 100)
        price_wide = prices2.pivot(index="date",columns="ticker",values="normalized").ffill()
        equal_weight = price_wide.mean(axis=1).reset_index(name="normalized")
        ecom2 = df_ecommerce.copy()
        ecom2["date"] = pd.to_datetime(ecom2["date"])
        ecom2 = ecom2[ecom2["date"].dt.year >= START_YEAR].copy()
        ecom2["norm"] = ecom2["index_level"]/ecom2["index_level"].iloc[0]*100

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(equal_weight["date"], equal_weight["normalized"],
                linewidth=2.5, color="steelblue",
                label=f"Equal-weighted Portfolio ({len(eligible)} companies)")
        ax.plot(ecom2["date"], ecom2["norm"], linewidth=2, linestyle="--",
                color="coral", label="E-commerce Index")
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), alpha=0.1, color="gray")
        ax.text(pd.Timestamp("2020-04-01"), equal_weight["normalized"].max()*0.9, "COVID-19", fontsize=9)
        ax.set_title("Growth of a $1 Equal-Weighted Investment in E-commerce Companies")
        ax.set_ylabel("Index Value (2015 = 100)")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.subheader("Bootstrap Correlation by Region")

        features = ["operating_margin","rd_intensity","free_cash_flow",
                    "total_assets","operating_cash_flow"]

        analysis_df = yearly.merge(df_fin, left_on=["ticker","year"],
                                   right_on=["ticker","fiscal_year"], how="left")
        analysis_df["fcf_margin"] = (analysis_df["free_cash_flow"] /
                                     analysis_df["total_revenue"])
        analysis_df["ocf_margin"] = (analysis_df["operating_cash_flow"] /
                                     analysis_df["total_revenue"])

        valid_features = [f for f in features if f in analysis_df.columns]
        region_counts = (analysis_df.groupby("region_x")["ticker"]
                         .nunique().sort_values(ascending=False))
        valid_regions = region_counts[region_counts>=4].index

        def bootstrap_corr(df, feats, n_boot=500):
            corr_sum = np.zeros((len(feats), len(feats)))
            for _ in range(n_boot):
                sample = df.sample(n=len(df), replace=True)
                corr_sum += sample[feats].corr(method="spearman").fillna(0).values
            return corr_sum / n_boot

        n_seg = len(valid_regions)
        if n_seg > 0:
            fig, axes = plt.subplots(n_seg, 1, figsize=(8, 5*n_seg))
            if n_seg == 1:
                axes = [axes]
            last_im = None
            for ax, region in zip(axes, valid_regions):
                subset = (analysis_df[analysis_df["region_x"]==region]
                          [valid_features].dropna())
                if len(subset)<5:
                    ax.axis("off")
                    continue
                corr_boot = bootstrap_corr(subset, valid_features, n_boot=500)
                im = ax.imshow(corr_boot, cmap="coolwarm", vmin=-1, vmax=1)
                last_im = im
                ax.set_xticks(range(len(valid_features)))
                ax.set_yticks(range(len(valid_features)))
                ax.set_xticklabels(valid_features, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(valid_features, fontsize=8)
                ax.set_title(f"{region}  (n={len(subset)})", fontsize=11)
                for i in range(len(valid_features)):
                    for j in range(len(valid_features)):
                        ax.text(j, i, f"{corr_boot[i,j]:.2f}", ha="center",
                                va="center", fontsize=8)
            fig.suptitle("Bootstrap Correlation Matrices by Region", fontsize=14, y=1.01)
            if last_im:
                fig.colorbar(last_im, ax=axes[-1], shrink=0.8, label="Bootstrap correlation")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════
# 7. HYPOTHESIS TESTING
# ══════════════════════════════════════════

elif page == "Hypothesis Testing":
    st.title("Hypothesis Testing")

    tab_h1, tab_h2 = st.tabs(["H1: Macro → CAGR", "H2: Volatility Developed vs Emerging"])

    # ── H1 ──
    with tab_h1:
        st.subheader("H1: Macroeconomic environment moderates post-IPO CAGR")
        st.markdown("""
        **Hypothesis:** Companies from countries with higher internet penetration, GDP per capita, 
        and FDI inflows demonstrate higher CAGR after IPO.

        **Method:** OLS regression with HC3 robust standard errors. Predictors are standardised 
        for coefficient comparability.
        """)

        macro_vars = ["internet_users_pct","gdp_per_capita_usd","fdi_inflow_pct_gdp"]
        df_h1 = df_companies[["cagr"]+macro_vars].dropna()

        if len(df_h1) > 10:
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(df_h1[macro_vars])
            X = sm.add_constant(X_sc)
            y = df_h1["cagr"]
            model = sm.OLS(y, X).fit(cov_type="HC3")

            col1, col2, col3 = st.columns(3)
            col1.metric("R²", f"{model.rsquared:.3f}")
            col2.metric("Adj R²", f"{model.rsquared_adj:.3f}")
            col3.metric("F-stat p-value", f"{model.f_pvalue:.3f}")

            st.code(str(model.summary()))

            # Scatter plots
            fig, axes = plt.subplots(1, 3, figsize=(15,5))
            labels = ["Internet Users %","GDP per Capita USD","FDI Inflow % GDP"]
            for ax, var, label in zip(axes, macro_vars, labels):
                df_p = df_companies[["cagr",var,"market_type"]].dropna()
                clr = df_p["market_type"].map({"Developed":"steelblue","Emerging":"coral"})
                ax.scatter(df_p[var], df_p["cagr"], c=clr, alpha=0.7, s=60, edgecolors="white")
                x_vals = df_p[var]; y_vals = df_p["cagr"]
                b_val, m_val = np.polyfit(x_vals, y_vals, 1)
                ax.plot(sorted(x_vals), [b_val+m_val*xi for xi in sorted(x_vals)],
                        color="black", linewidth=1.5, linestyle="--", alpha=0.7)
                r, p = stats.pearsonr(x_vals, y_vals)
                ax.set_title(f"{label}\nr={r:.3f}, p={p:.3f}", fontsize=10)
                ax.set_xlabel(label, fontsize=9)
                ax.set_ylabel("CAGR" if ax==axes[0] else "")
                ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
            from matplotlib.patches import Patch
            axes[2].legend(handles=[Patch(facecolor="steelblue",label="Developed"),
                                    Patch(facecolor="coral",label="Emerging")], fontsize=9)
            fig.suptitle("CAGR vs Macro Indicators (H1)", fontsize=13, fontweight="bold")
            st.pyplot(fig)
            plt.close()

            st.markdown("""
            **Result:** H1 is **not confirmed**. The model R²=0.076 with F-stat p=0.664 is statistically 
            insignificant. None of the three macro predictors reach p<0.05. Macroeconomic environment 
            at the country level does not explain post-IPO CAGR in this sample.
            """)

    # ── H2 ──
    with tab_h2:
        st.subheader("H2: Emerging markets show higher post-IPO volatility")
        st.markdown("""
        **Hypothesis:** Companies from emerging markets exhibit higher annualized volatility 
        after IPO compared to developed markets.

        **Method:** Permutation test (10,000 permutations) on median rolling volatility 
        per company (252-day window). Effect size measured by Cohen's d.
        """)

        vol_per_company = (
            df_stock_copy.groupby(["company_id","market_type"])["rolling_vol"]
            .median().reset_index().dropna()
        )

        if len(vol_per_company) > 5:
            developed = vol_per_company[vol_per_company["market_type"]=="Developed"]["rolling_vol"]
            emerging  = vol_per_company[vol_per_company["market_type"]=="Emerging"]["rolling_vol"]

            observed_diff = emerging.mean() - developed.mean()
            combined = np.concatenate([developed, emerging])
            n_dev = len(developed)
            perm_diffs = []
            np.random.seed(42)
            for _ in range(10000):
                shuffled = np.random.permutation(combined)
                perm_diffs.append(shuffled[:n_dev].mean() - shuffled[n_dev:].mean())
            p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
            d = cohen_d(emerging, developed)
            effect = "small" if abs(d)<0.2 else "medium" if abs(d)<0.8 else "large"

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Emerging mean vol", f"{emerging.mean():.4f}")
            col2.metric("Developed mean vol", f"{developed.mean():.4f}")
            col3.metric("p-value (permutation)", f"{p_value:.4f}")
            col4.metric(f"Cohen's d ({effect})", f"{d:.3f}")

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            sns.violinplot(data=vol_per_company, x="market_type", y="rolling_vol",
                           inner="box", ax=axes[0],
                           palette={"Developed":"steelblue","Emerging":"coral"})
            for i, (mtype, grp) in enumerate(vol_per_company.groupby("market_type")["rolling_vol"]):
                axes[0].scatter(i, grp.mean(), color="black", zorder=5, s=80, marker="D")
            axes[0].set_title(f"Median Rolling Volatility per Company\np={p_value:.4f} | d={d:.3f}")
            axes[0].set_ylabel("Median Rolling Volatility")
            axes[0].text(0.98, 0.95, f"Effect: {effect}", transform=axes[0].transAxes,
                         fontsize=9, ha="right", color="gray")

            vol_by_year = (
                df_stock_copy.groupby(["year","market_type"])["rolling_vol"]
                .median().reset_index().dropna()
            )
            for mtype, color in [("Developed","steelblue"),("Emerging","coral")]:
                data = vol_by_year[vol_by_year["market_type"]==mtype]
                axes[1].plot(data["year"], data["rolling_vol"],
                             color=color, linewidth=2, marker="o", label=mtype)
            axes[1].axvline(2020, color="gray", linestyle="--", alpha=0.5, label="COVID-19")
            axes[1].axvline(2022, color="orange", linestyle="--", alpha=0.5, label="Rate hikes")
            axes[1].set_title("Rolling Volatility Over Time")
            axes[1].set_xlabel("Year")
            axes[1].set_ylabel("Median Rolling Volatility")
            axes[1].legend()

            st.pyplot(fig)
            plt.close()

            st.markdown(f"""
            **Result:** H2 is **not confirmed** (p={p_value:.4f} > 0.05). The direction of the effect 
            is **opposite** to the hypothesis — developed-market companies show *higher* volatility 
            (Cohen's d={d:.3f}, {effect} effect). This likely reflects the dominance of high-volatility 
            North American tech companies in the developed subsample. The time-series panel shows both 
            groups spike during COVID-19 (2020–2021) and rate-hike periods (2022), with no systematic 
            divergence.
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
