import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("Customer Segmentation Dashboard")
st.write("Enter customer information to predict the customer cluster.")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    kmeans = pickle.load(open("kmeans_model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    pca = pickle.load(open("pca.pkl","rb"))
    return kmeans, scaler, pca

kmeans, scaler, pca = load_models()

@st.cache_data
def load_data():
    return pd.read_csv("pca_data.csv")

pca_data = load_data()

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Customer Details")

income = st.sidebar.number_input("Income", 0, 200000, 50000)
age = st.sidebar.number_input("Age", 18, 100, 30)
recency = st.sidebar.number_input("Recency (days)", 0, 100, 20)

total_spending = st.sidebar.number_input("Total Spending", 0, 10000, 500)

web_purchases = st.sidebar.number_input("Web Purchases", 0, 50, 5)
store_purchases = st.sidebar.number_input("Store Purchases", 0, 50, 5)
catalog_purchases = st.sidebar.number_input("Catalog Purchases", 0, 50, 2)
web_visits = st.sidebar.number_input("Web Visits per Month", 0, 50, 10)

# -------- UI ONLY (NOT USED IN MODEL) --------
marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Widow"]
)

education = st.sidebar.selectbox(
    "Education",
    ["Basic", "Graduation", "Master", "PhD"]
)

predict = st.sidebar.button("Predict Cluster")

# ---------------- PREDICTION ----------------
if predict:

    input_data = pd.DataFrame({
        "Income":[income],
        "Age":[age],
        "Recency":[recency],
        "Total_Spending":[total_spending],
        "NumWebPurchases":[web_purchases],
        "NumStorePurchases":[store_purchases],
        "NumCatalogPurchases":[catalog_purchases],
        "NumWebVisitsMonth":[web_visits]
    })

    st.subheader("Customer Input Data")
    st.write(input_data)

    # -------- SCALE DATA --------
    scaled = scaler.transform(input_data)

    # -------- PREDICT --------
    cluster = kmeans.predict(scaled)[0]

    st.subheader("Predicted Customer Segment")
    st.success(f"This customer belongs to Cluster {cluster}")

    # -------- PCA --------
    pca_point = pca.transform(scaled)

    pc1 = pca_point[0][0]
    pc2 = pca_point[0][1]

    # -------- VISUALIZATION --------
    st.subheader("Customer Segmentation Visualization")

    fig = px.scatter(
        pca_data,
        x="PC1",
        y="PC2",
        color=pca_data["Cluster"].astype(str),
        title="Customer Clusters (PCA)"
    )

    fig.add_scatter(
        x=[pc1],
        y=[pc2],
        mode="markers",
        marker=dict(size=15, color="black"),
        name="New Customer"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------- CLUSTER DISTRIBUTION --------
    st.subheader("Cluster Distribution")

    cluster_counts = pca_data["Cluster"].value_counts().sort_index()

    st.bar_chart(cluster_counts)