import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 End-to-End Sales Forecasting & Demand Intelligence System")
st.markdown("---")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y",
        errors="coerce"
)

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%d/%m/%Y",
        errors="coerce"
)

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Week"] = df["Order Date"].dt.isocalendar().week
    df["Day"] = df["Order Date"].dt.day_name()

    return df

df = load_data()

df.dropna(subset=["Order Date"], inplace=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Sales Overview",
        "📈 Forecast Explorer",
        "🚨 Anomaly Report",
        "📦 Product Demand Segments"
    ]
)

st.sidebar.markdown("---")

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Year"].unique())
)

selected_region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + sorted(df["Region"].unique())
)

selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(df["Category"].unique())
)

filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df["Year"] == selected_year
]

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == selected_region
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == selected_category
    ]

# -----------------------------
# KPI Cards
# -----------------------------
if page == "🏠 Sales Overview":

    total_sales = filtered_df["Sales"].sum()

    total_orders = filtered_df.shape[0]

    total_categories = filtered_df["Category"].nunique()

    total_regions = filtered_df["Region"].nunique()

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "📦 Orders",
        total_orders
    )

    c3.metric(
        "🛍 Categories",
        total_categories
    )

    c4.metric(
        "🌍 Regions",
        total_regions
    )

    st.markdown("---")

    # ------------------------------------
    # Yearly Sales
    # ------------------------------------

    yearly = df.groupby("Year")["Sales"].sum().reset_index()

    fig = px.bar(
        yearly,
        x="Year",
        y="Sales",
        title="Yearly Sales"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # ------------------------------------
    # Monthly Sales Trend
    # ------------------------------------

    monthly = filtered_df.groupby("Month")["Sales"].sum().reset_index()

    fig = px.line(
        monthly,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # ------------------------------------
    # Region vs Sales
    # ------------------------------------

    c1,c2 = st.columns(2)

    with c1:

        region_sales = filtered_df.groupby(
            "Region"
        )["Sales"].sum().reset_index()

        fig = px.bar(
            region_sales,
            x="Region",
            y="Sales",
            color="Region",
            title="Sales by Region"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with c2:

        category_sales = filtered_df.groupby(
            "Category"
        )["Sales"].sum().reset_index()

        fig = px.pie(
            category_sales,
            names="Category",
            values="Sales",
            title="Sales by Category"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.markdown("### Sample Data")

    st.dataframe(
        filtered_df.head(20),
        width="stretch"
    )
# ------------------------------------------------------------
# Forecast Explorer
# ------------------------------------------------------------

elif page == "📈 Forecast Explorer":

    st.header("📈 Sales Forecast Explorer")

    category = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    region = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    horizon = st.slider(
        "Forecast Horizon (Months)",
        1,
        3,
        3
    )

    forecast_df = df[
        (df["Category"] == category) &
        (df["Region"] == region)
    ].copy()

    monthly_sales = forecast_df.groupby(
        pd.Grouper(key="Order Date", freq="ME")
    )["Sales"].sum().reset_index()

    st.subheader("Historical Monthly Sales")

    fig = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Historical Sales"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # ---------------------------------------------
    # Moving Average Forecast
    # ---------------------------------------------

    forecast = monthly_sales.copy()

    future_dates = pd.date_range(
        start=forecast["Order Date"].max() + pd.offsets.MonthEnd(1),
        periods=horizon,
        freq="ME"
    )

    future_values = []

    sales = list(forecast["Sales"])

    for i in range(horizon):

        prediction = np.mean(sales[-3:])

        future_values.append(prediction)

        sales.append(prediction)

    future = pd.DataFrame({

        "Order Date": future_dates,

        "Sales": future_values

    })

    st.subheader("Forecast")

    fig = px.line()

    fig.add_scatter(
        x=forecast["Order Date"],
        y=forecast["Sales"],
        mode="lines+markers",
        name="Historical"
    )

    fig.add_scatter(
        x=future["Order Date"],
        y=future["Sales"],
        mode="lines+markers",
        name="Forecast"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # ---------------------------------------------
    # Forecast Table
    # ---------------------------------------------

    st.subheader("Forecast Values")

    display = future.copy()

    display.columns = [
        "Month",
        "Forecast Sales"
    ]

    st.dataframe(
        display,
        width="stretch"
    )

    # ---------------------------------------------
    # Forecast Metrics
    # ---------------------------------------------

    c1, c2, c3 = st.columns(3)

    mae = np.mean(
        np.abs(
            monthly_sales["Sales"] -
            monthly_sales["Sales"].rolling(3).mean().fillna(
                monthly_sales["Sales"].mean()
            )
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                monthly_sales["Sales"] -
                monthly_sales["Sales"].rolling(3).mean().fillna(
                    monthly_sales["Sales"].mean()
                )
            )**2
        )
    )

    c1.metric(
        "MAE",
        f"{mae:.2f}"
    )

    c2.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    c3.metric(
        "Forecast Months",
        horizon
    )

    st.success(
        "The forecast above is generated using a Moving Average baseline suitable for dashboard visualization. Your notebook contains the full SARIMA, Prophet, and XGBoost models for detailed forecasting."
    )
# ------------------------------------------------------------
# Anomaly Report
# ------------------------------------------------------------

elif page == "🚨 Anomaly Report":

    st.header("🚨 Weekly Sales Anomaly Detection")

    weekly = df.groupby(
        pd.Grouper(
            key="Order Date",
            freq="W"
        )
    )["Sales"].sum().reset_index()

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly["Anomaly"] = model.fit_predict(
        weekly[["Sales"]]
    )

    anomalies = weekly[
        weekly["Anomaly"] == -1
    ]

    fig = px.line(
        weekly,
        x="Order Date",
        y="Sales",
        title="Weekly Sales"
    )

    fig.add_scatter(
        x=anomalies["Order Date"],
        y=anomalies["Sales"],
        mode="markers",
        marker=dict(
            size=10,
            color="red"
        ),
        name="Anomaly"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomalies,
        width="stretch"
    )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Total Weeks",
        len(weekly)
    )

    c2.metric(
        "Anomalies",
        len(anomalies)
    )

    c3.metric(
        "Anomaly %",
        f"{len(anomalies)/len(weekly)*100:.2f}%"
    )

    st.info(
        "Red points represent unusual sales behaviour detected using Isolation Forest."
    )

# ------------------------------------------------------------
# Product Demand Segments
# ------------------------------------------------------------

else:

    st.header("📦 Product Demand Segmentation")

    product_data = df.groupby(
        "Sub-Category"
    ).agg(

        TotalSales=("Sales","sum"),
        AvgSales=("Sales","mean"),
        SalesStd=("Sales","std"),
        OrderCount=("Sales","count")

    ).reset_index()

    product_data.fillna(0,inplace=True)

    features = product_data[
        [
            "TotalSales",
            "AvgSales",
            "SalesStd",
            "OrderCount"
        ]
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init="10"
    )

    product_data["Cluster"] = model.fit_predict(scaled)

    pca = PCA(
        n_components=2
    )

    pcs = pca.fit_transform(scaled)

    product_data["PC1"] = pcs[:,0]
    product_data["PC2"] = pcs[:,1]

    cluster_names = {

        0:"High Demand",

        1:"Growing Demand",

        2:"Low Demand",

        3:"Seasonal Demand"

    }

    product_data["Demand Segment"] = product_data[
        "Cluster"
    ].map(cluster_names)

    fig = px.scatter(

        product_data,

        x="PC1",

        y="PC2",

        color="Demand Segment",

        text="Sub-Category",

        title="Product Demand Clusters"

    )

    fig.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("Cluster Summary")

    st.dataframe(

        product_data[
            [

                "Sub-Category",

                "TotalSales",

                "AvgSales",

                "OrderCount",

                "Demand Segment"

            ]

        ],

        width="stretch"

    )

    st.markdown("---")

    st.subheader("📋 Business Recommendations")

    col1,col2 = st.columns(2)

    with col1:

        st.success(
            """
High Demand

• Maintain higher inventory

• Prioritize fast delivery

• Avoid stock-outs
"""
        )

        st.warning(
            """
Seasonal Demand

• Stock before peak season

• Monitor monthly trend

• Plan promotional campaigns
"""
        )

    with col2:

        st.info(
            """
Growing Demand

• Increase inventory gradually

• Monitor customer demand

• Expand marketing
"""
        )

        st.error(
            """
Low Demand

• Reduce inventory

• Bundle with other products

• Avoid overstock
"""
        )

    st.markdown("---")

    st.subheader("Demand Segment Distribution")

    count_df = product_data["Demand Segment"].value_counts().reset_index()

    count_df.columns = [

        "Demand Segment",

        "Products"

    ]

    fig = px.bar(

        count_df,

        x="Demand Segment",

        y="Products",

        color="Demand Segment",

        title="Products per Demand Segment"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )
# ============================================================
# PROFESSIONAL DASHBOARD FOOTER
# ============================================================

st.markdown("---")

st.markdown(
"""
<style>

.metric-box{
    background-color:#f5f7fa;
    padding:15px;
    border-radius:10px;
    border:1px solid #d3d3d3;
}

.footer{
    text-align:center;
    padding:20px;
    font-size:15px;
    color:gray;
}

.big-title{
    font-size:30px;
    font-weight:bold;
}

.small{
    color:gray;
}

</style>
""",
unsafe_allow_html=True
)

st.markdown("---")

st.subheader("📊 Dashboard Summary")

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Dataset Rows",
    f"{len(df):,}"
)

c2.metric(
    "Total Sales",
    f"${df['Sales'].sum():,.0f}"
)

c3.metric(
    "Categories",
    df["Category"].nunique()
)

c4.metric(
    "Regions",
    df["Region"].nunique()
)

st.markdown("---")

st.subheader("📌 Project Highlights")

st.markdown("""

✅ Time Series Analysis

✅ Sales Trend Visualization

✅ Forecast Explorer

✅ Anomaly Detection

✅ Product Demand Segmentation

✅ Interactive Dashboard

✅ Business Insights

""")

st.markdown("---")

st.subheader("💼 Business Recommendations")

st.success("""
✔ Maintain sufficient inventory for High Demand products.
""")

st.info("""
✔ Increase stock gradually for Growing Demand products.
""")

st.warning("""
✔ Prepare inventory before Seasonal Demand periods.
""")

st.error("""
✔ Reduce stock levels for consistently Low Demand products.
""")

st.markdown("---")

st.subheader("📄 Download Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name="SalesForecast_Data.csv",
    mime="text/csv"
)

st.markdown("---")

st.subheader("ℹ About this Dashboard")

st.write("""
This dashboard was developed as part of the **End-to-End Sales Forecasting & Demand Intelligence System** internship project.

It demonstrates:

- Sales analytics
- Time series forecasting
- Demand segmentation
- Anomaly detection
- Business intelligence
- Interactive visualization using Streamlit
""")

st.markdown("---")

st.markdown(
"""
<div class="footer">

<b>End-to-End Sales Forecasting & Demand Intelligence System</b>

<br>

Developed using Python • Pandas • Plotly • Scikit-Learn • Streamlit

<br><br>

© 2026 Internship Project

</div>
""",
unsafe_allow_html=True
)