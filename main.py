import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from io import BytesIO
import time

# Initialize directories
os.makedirs("data", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

# Streamlit configuration
st.set_page_config(
    page_title="AI-Powered Data Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App Header
st.markdown("<h1 style='text-align: center;'>🌟 AI-Powered Data Cleaning & Analysis Tool</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>An intelligent platform for data professionals!</h3>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🔧 Tools Menu")
menu = st.sidebar.radio(
    "Choose a Tool",
    [
        "🏠 Home",
        "📤 Upload Data",
        "🛠️ Data Cleaning",
        "📊 Data Visualization",
        "🛠️ Pivot Table",
        "🤖 Machine Learning Tools",
        "🛒 Market Basket Analysis",
        "📈 Time Series Forecasting",
        "📉 Stock Market Analysis",
        "📂 File Manager",
    ]
)

# Animation Example for Home Page
if menu == "🏠 Home":
    st.header("Welcome to the AI-Powered Data Tool!")
    st.markdown(
        """
        This tool offers an array of features for data cleaning, analysis, and visualization:
        - Data Cleaning & Processing
        - Advanced Visualizations
        - Machine Learning Analysis
        - Market Basket Analysis
        - Time Series Forecasting
        - Stock Market Analysis
        
        Use the menu on the left to navigate between tools.
        """
    )
    st.markdown("<h3 style='text-align: center;'>✨ Let’s make data powerful together! ✨</h3>", unsafe_allow_html=True)
    st.balloons()
    time.sleep(3)

# File Upload Functionality
elif menu == "📤 Upload Data":
    st.header("📤 Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel files", type=["csv", "xlsx"])
    if uploaded_file:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

# Data Cleaning Functionality
elif menu == "🛠️ Data Cleaning":
    st.header("🛠️ Data Cleaning Tools")
    uploaded_files = os.listdir("data/")

    if uploaded_files:
        selected_file = st.selectbox("Select a file for cleaning", uploaded_files)
        file_path = os.path.join("data", selected_file)
        df = pd.read_csv(file_path) if selected_file.endswith(".csv") else pd.read_excel(file_path)

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df)

            # Cleaning Options
            if st.checkbox("Remove Duplicates"):
                df = df.drop_duplicates()
                st.success("Duplicates removed!")
            if st.checkbox("Handle Missing Values"):
                strategy = st.selectbox("Select a strategy", ["Mean", "Median", "Mode", "Drop Rows"])
                if strategy == "Mean":
                    df.fillna(df.mean(), inplace=True)
                elif strategy == "Median":
                    df.fillna(df.median(), inplace=True)
                elif strategy == "Mode":
                    df.fillna(df.mode().iloc[0], inplace=True)
                elif strategy == "Drop Rows":
                    df.dropna(inplace=True)
                st.success("Missing values handled!")

            # Save cleaned data
            save_path = os.path.join("processed_data", f"cleaned_{selected_file}")
            df.to_csv(save_path, index=False)
            st.download_button(
                label="📥 Download Cleaned Data",
                data=open(save_path, "rb").read(),
                file_name=f"cleaned_{selected_file}"
            )

# Data Visualization Functionality
elif menu == "📊 Data Visualization":
    st.header("📊 Data Visualization")
    uploaded_files = os.listdir("data/")

    if uploaded_files:
        selected_file = st.selectbox("Select a file for visualization", uploaded_files)
        file_path = os.path.join("data", selected_file)
        df = pd.read_csv(file_path) if selected_file.endswith(".csv") else pd.read_excel(file_path)

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df)

            chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Scatter", "Pie", "Boxplot", "Histogram", "Heatmap"])
            x_axis = st.selectbox("X-axis", df.columns)
            y_axis = st.selectbox("Y-axis", df.columns)

            try:
                if chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis, title="Line Chart")
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, title="Bar Chart")
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, title="Scatter Plot")
                elif chart_type == "Pie":
                    fig = px.pie(df, names=x_axis, title="Pie Chart")
                elif chart_type == "Boxplot":
                    fig = px.box(df, x=x_axis, y=y_axis, title="Boxplot")
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, title="Histogram")
                elif chart_type == "Heatmap":
                    corr_matrix = df.corr()
                    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='Viridis'))
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error occurred: {e}")

# Pivot Table Functionality
elif menu == "🛠️ Pivot Table":
    st.header("🛠️ Pivot Table Generation")
    uploaded_files = os.listdir("data/")

    if uploaded_files:
        selected_file = st.selectbox("Select a file for Pivot Table", uploaded_files)
        file_path = os.path.join("data", selected_file)
        df = pd.read_csv(file_path) if selected_file.endswith(".csv") else pd.read_excel(file_path)

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df)

            try:
                index_column = st.selectbox("Select Index Column", df.columns)
                columns_column = st.selectbox("Select Columns Column", df.columns)
                values_column = st.selectbox("Select Values Column", df.columns)
                agg_function = st.selectbox("Select Aggregation Function", ["sum", "mean", "count", "max", "min"])

                if index_column and columns_column and values_column:
                    pivot_table = pd.pivot_table(
                        df,
                        index=[index_column],
                        columns=[columns_column],
                        values=[values_column],
                        aggfunc=agg_function,
                    )

                    st.subheader("Pivot Table")
                    st.dataframe(pivot_table)

            except Exception as e:
                st.error(f"Error occurred while generating Pivot Table: {e}")

# Machine Learning Tools
elif menu == "🤖 Machine Learning Tools":
    st.header("🤖 Machine Learning Tools")
    uploaded_files = os.listdir("data/")

    if uploaded_files:
        selected_file = st.selectbox("Select a file for ML analysis", uploaded_files)
        file_path = os.path.join("data", selected_file)
        df = pd.read_csv(file_path) if selected_file.endswith(".csv") else pd.read_excel(file_path)

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df)

            # Linear Regression Example
            if st.checkbox("Perform Linear Regression"):
                target = st.selectbox("Select Target Column", df.columns)
                features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target])

                if target and features:
                    X = df[features]
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    st.write("R-squared:", r2_score(y_test, predictions))
                    st.write("MSE:", mean_squared_error(y_test, predictions))

# Market Basket Analysis
elif menu == "🛒 Market Basket Analysis":
    st.header("🛒 Market Basket Analysis")
    uploaded_files = os.listdir("data/")

    if uploaded_files:
        selected_file = st.selectbox("Select a file for market basket analysis", uploaded_files)
        file_path = os.path.join("data", selected_file)
        df = pd.read_csv(file_path) if selected_file.endswith(".csv") else pd.read_excel(file_path)

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df)

            # Generate frequent itemsets and association rules
            if st.checkbox("Generate Association Rules"):
                min_support = st.slider("Select Minimum Support", 0.01, 0.5, 0.1)
                frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

                st.subheader("Frequent Itemsets")
                st.dataframe(frequent_itemsets)
                st.subheader("Association Rules")
                st.dataframe(rules)

# Time Series Forecasting
elif menu == "📈 Time Series Forecasting":
    st.header("📈 Time Series Forecasting")
    uploaded_files = os.listdir("data/")

    if uploaded_files:
        selected_file = st.selectbox("Select a file for time series forecasting", uploaded_files)
        file_path = os.path.join("data", selected_file)
        df = pd.read_csv(file_path) if selected_file.endswith(".csv") else pd.read_excel(file_path)

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df)

            # Time series forecasting with ARIMA
            column = st.selectbox("Select Time Series Column", df.columns)
            if column:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                df.dropna(inplace=True)

                model = ARIMA(df[column], order=(5, 1, 0))
                results = model.fit()

                st.subheader("Forecast Results")
                st.line_chart(results.fittedvalues)

# Stock Market Analysis
elif menu == "📉 Stock Market Analysis":
    st.header("📉 Stock Market Analysis")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL for Apple)")
    if ticker:
        data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
        st.write("Stock Data")
        st.dataframe(data)

        st.write("Closing Prices")
        st.line_chart(data["Close"])

# File Manager
elif menu == "📂 File Manager":
    st.header("📂 File Manager")
    uploaded_files = os.listdir("data/")

    if uploaded_files:
        for file in uploaded_files:
            col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
            with col1:
                st.text(file)
            with col2:
                if st.button(f"📂 View {file}"):
                    file_path = os.path.join("data", file)
                    st.write(pd.read_csv(file_path).head())
            with col3:
                if st.button(f"❌ Delete {file}"):
                    os.remove(os.path.join("data", file))
                    st.success(f"Deleted {file}")
            with col4:
                if st.button(f"🔗 Share {file}"):
                    st.write(f"Shared link: [localhost/data/{file}](#)")
