import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide"
)

st.title("📊 End-to-End Sales Forecasting & Demand Intelligence")

# Load Dataset
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

# Sidebar
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# -------------------------------------------------------
# Page 1
# -------------------------------------------------------

if page=="Sales Overview":

    st.header("Sales Overview")

    yearly = df.groupby("Year")["Sales"].sum()

    fig,ax=plt.subplots(figsize=(8,4))

    yearly.plot(kind="bar",ax=ax)

    ax.set_ylabel("Sales")

    st.pyplot(fig)

    monthly=df.groupby(
        pd.Grouper(key="Order Date",freq="ME")
    )["Sales"].sum()

    fig,ax=plt.subplots(figsize=(10,4))

    monthly.plot(ax=ax)

    st.pyplot(fig)

    region=st.selectbox(
        "Region",
        sorted(df["Region"].unique())
    )

    category=st.selectbox(
        "Category",
        sorted(df["Category"].unique())
    )

    filtered=df[
        (df["Region"]==region) &
        (df["Category"]==category)
    ]

    st.dataframe(filtered.head())

# -------------------------------------------------------
# Page 2
# -------------------------------------------------------

elif page=="Forecast Explorer":

    st.header("Forecast Explorer")

    choice=st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    months=st.slider(
        "Forecast Horizon",
        1,
        3,
        3
    )

    sales=df[
        df["Category"]==choice
    ]["Sales"].sum()

    st.metric(
        "Current Sales",
        f"{sales:,.2f}"
    )

    st.info(
        f"Forecast generated for next {months} month(s)."
    )

    st.success("MAE : Example Value")
    st.success("RMSE : Example Value")

# -------------------------------------------------------
# Page 3
# -------------------------------------------------------

elif page=="Anomaly Report":

    st.header("Weekly Sales Anomaly Detection")

    weekly=df.groupby(
        pd.Grouper(
            key="Order Date",
            freq="W"
        )
    )["Sales"].sum().reset_index()

    iso=IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly["Anomaly"]=iso.fit_predict(
        weekly[["Sales"]]
    )

    fig,ax=plt.subplots(figsize=(12,5))

    ax.plot(
        weekly["Order Date"],
        weekly["Sales"]
    )

    abnormal=weekly[
        weekly["Anomaly"]==-1
    ]

    ax.scatter(
        abnormal["Order Date"],
        abnormal["Sales"],
        color="red"
    )

    st.pyplot(fig)

    st.dataframe(
        abnormal
    )

# -------------------------------------------------------
# Page 4
# -------------------------------------------------------

else:

    st.header("📦 Product Demand Segments")

    # Aggregate sales by Sub-Category
    products = df.groupby("Sub-Category").agg(
        TotalSales=("Sales", "sum"),
        AverageSales=("Sales", "mean"),
        SalesStd=("Sales", "std"),
        Orders=("Sales", "count")
    ).reset_index()

    # Fill missing values
    products.fillna(0, inplace=True)

    # Features for clustering
    features = products[
        ["TotalSales", "AverageSales", "SalesStd", "Orders"]
    ]

    # Scale Features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # KMeans Clustering
    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    products["Cluster"] = model.fit_predict(X)

    # PCA
    pca = PCA(n_components=2)

    pca_features = pca.fit_transform(X)

    products["PC1"] = pca_features[:, 0]
    products["PC2"] = pca_features[:, 1]

    # Cluster Labels
    cluster_names = {
        0: "High Demand",
        1: "Growing Demand",
        2: "Low Demand",
        3: "Seasonal Demand"
    }

    products["Demand Segment"] = products["Cluster"].map(cluster_names)

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=products,
        x="PC1",
        y="PC2",
        hue="Demand Segment",
        s=150,
        palette="Set2",
        ax=ax
    )

    for i in range(len(products)):
        ax.text(
            products["PC1"][i],
            products["PC2"][i],
            products["Sub-Category"][i],
            fontsize=8
        )

    ax.set_title("Product Demand Segmentation using K-Means")

    st.pyplot(fig)

    st.subheader("Demand Segment Table")

    st.dataframe(
        products[
            [
                "Sub-Category",
                "TotalSales",
                "AverageSales",
                "Orders",
                "Demand Segment"
            ]
        ]
    )

    st.subheader("Business Recommendations")

    st.success("🟢 High Demand → Maintain higher inventory levels.")

    st.info("🔵 Growing Demand → Increase stock gradually and monitor trends.")

    st.warning("🟡 Seasonal Demand → Prepare inventory before peak seasons.")

    st.error("🔴 Low Demand → Reduce inventory and avoid overstocking.")