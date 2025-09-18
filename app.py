import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

# =============================
# Load models & encoders
# =============================
models = {
    "Linear Regression": joblib.load("linear.pkl"),
    "Random Forest": joblib.load("rf.pkl"),
    "XGBoost": joblib.load("xgb.pkl")
}

encoders = joblib.load("encoders.pkl")  # dictionary of fitted LabelEncoders

st.set_page_config(page_title="🌾 Market Price AI Platform", layout="wide")
st.title("🌾 AI-driven Market Access & Price Predictor")
st.markdown("Connect farmers to buyers, analyze markets, and predict prices.")

# =============================
# Dataset Upload & EDA
# =============================
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Market Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 🔥 Clean column names
    df.rename(columns={
        "Min_x0020_Price": "Min_Price",
        "Max_x0020_Price": "Max_Price",
        "Modal_x0020_Price": "Modal_Price"
    }, inplace=True)

    # ✅ Ensure Arrival_Date is datetime
    if "Arrival_Date" in df.columns:
        df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")
        df = df.dropna(subset=["Arrival_Date"])

    st.write("### Preview of Uploaded Data", df.head())

    # --- Filters for interactive EDA ---
    st.sidebar.header("🔎 Filters")
    sel_state = st.sidebar.selectbox("Select State", ["All"] + sorted(df["State"].unique().tolist()))
    sel_comm = st.sidebar.selectbox("Select Commodity", ["All"] + sorted(df["Commodity"].unique().tolist()))

    df_filtered = df.copy()
    if sel_state != "All":
        df_filtered = df_filtered[df_filtered["State"] == sel_state]
    if sel_comm != "All":
        df_filtered = df_filtered[df_filtered["Commodity"] == sel_comm]

    # --- EDA Visuals ---
    st.subheader("📊 Market Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Top 10 Commodities by Frequency**")
        fig, ax = plt.subplots(figsize=(6, 4))
        df_filtered["Commodity"].value_counts().head(10).plot(kind="bar", color="skyblue", ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("**State-wise Average Modal Price**")
        fig, ax = plt.subplots(figsize=(6, 4))
        state_prices = df_filtered.groupby("State")["Modal_Price"].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=state_prices.index, y=state_prices.values, ax=ax, palette="magma")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
        # st.write("**Average Modal Price by State**")
        # fig, ax = plt.subplots(figsize=(10, 5))
        # avg_state = df_filtered.groupby("State")["Modal_Price"].mean().sort_values(ascending=False)
        # sns.barplot(x=avg_state.index, y=avg_state.values, ax=ax, palette="coolwarm")
        # plt.xticks(rotation=45)
        # st.pyplot(fig)
        
        # st.write("**📅 Month-wise Average Modal Price**")
        # df_filtered["Month"] = df_filtered["Arrival_Date"].dt.month
        # monthly_avg = df_filtered.groupby("Month")["Modal_Price"].mean()
        # fig, ax = plt.subplots(figsize=(6, 4))
        # monthly_avg.plot(kind="bar", ax=ax, color="coral")
        # st.pyplot(fig)
        
        # st.write("**📅 Month-wise Average Modal Price**")
        # df_filtered["Month"] = df_filtered["Arrival_Date"].dt.month
        # monthly_avg = df_filtered.groupby("Month")["Modal_Price"].mean()
        # fig, ax = plt.subplots(figsize=(6, 4))
        # monthly_avg.plot(kind="bar", ax=ax, color="coral")
        # st.pyplot(fig)

        
        # st.write("**📈 Trend of Modal Price Over Time**")
        # fig, ax = plt.subplots(figsize=(8, 4))
        # df_filtered.groupby("Arrival_Date")["Modal_Price"].mean().plot(ax=ax, color="green")
        # ax.set_ylabel("Average Modal Price")
        # ax.set_xlabel("Date")
        # st.pyplot(fig)



# =============================
# Prediction Section
# =============================
st.subheader("🔮 Predict Modal Price")

if uploaded_file:   # ✅ ensures df exists
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        # Dropdowns from dataset unique values
        state = col1.selectbox("State", sorted(df["State"].dropna().unique()))
        commodity = col2.selectbox("Commodity", sorted(df["Commodity"].dropna().unique()))
        district = col3.selectbox("District", sorted(df["District"].dropna().unique()))

        market = col1.selectbox("Market", sorted(df["Market"].dropna().unique()))
        variety = col2.selectbox("Variety", sorted(df["Variety"].dropna().unique()))
        grade = col3.selectbox("Grade", sorted(df["Grade"].dropna().unique()))

        min_price = col1.number_input("Min Price", min_value=0.0, step=1.0)
        max_price = col2.number_input("Max Price", min_value=0.0, step=1.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Apply same encoders as training
        def encode_value(col, val):
            if col in encoders and val in encoders[col].classes_:
                return encoders[col].transform([val])[0]
            else:
                st.warning(f"⚠️ '{val}' not seen in training for column {col}. Using 0 as fallback.")
                return 0

        input_data = pd.DataFrame([{
            "State": encode_value("State", state),
            "District": encode_value("District", district),
            "Market": encode_value("Market", market),
            "Commodity": encode_value("Commodity", commodity),
            "Variety": encode_value("Variety", variety),
            "Grade": encode_value("Grade", grade),
            "Min_Price": min_price,
            "Max_Price": max_price,
            "Year": 2025,
            "Month": 9,
            "Day": 17,
            "Weekday": 2
        }])

        st.write("### Encoded Input Features", input_data)

        # Predictions
        preds = {name: model.predict(input_data)[0] for name, model in models.items()}

        st.subheader("📈 Model Predictions")
        results_df = pd.DataFrame(list(preds.items()), columns=["Model", "Predicted Modal Price"])
        st.table(results_df)

        # Bar plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Model", y="Predicted Modal Price", data=results_df, palette="viridis", ax=ax)
        st.pyplot(fig)

        # Download option
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

else:
    st.warning("⚠️ Please upload a dataset to enable prediction form.")

# =============================
# Performance Dashboard (Optional)
# =============================
st.sidebar.header("📊 Model Performance")
if st.sidebar.checkbox("Show Training Metrics"):
    try:
        metrics = pd.read_csv("model_metrics.csv")  # Save during training
        st.sidebar.write(metrics)
    except:
        st.sidebar.warning("No metrics file found. Save evaluation results during training.")
