
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("credit_data.csv")
    return df

# Load model (for demonstration, will simulate here)
@st.cache_resource
def load_model():
    model = RandomForestClassifier()
    df = load_data()
    features = df[["Age", "Gender", "Education_Level", "Farm_Size_HA", "Years_of_Experience",
                   "Previous_Loan", "Mobile_Money_Use", "Market_Access",
                   "Average_Annual_Yield_Tons", "Income_per_Year_NGN", "Repayment_History_Score"]]
    target = df["Risk_Tier"]
    model.fit(features, target)
    return model

# App layout
st.title("Farmer Credit Risk Dashboard")
st.markdown("Analyze agripreneur data and predict credit risk tiers.")

df = load_data()
model = load_model()

# Sidebar for filters
st.sidebar.header("Filter by Risk Tier")
tiers = st.sidebar.multiselect("Select Tiers:", options=df["Risk_Tier"].unique(), default=df["Risk_Tier"].unique())
filtered_df = df[df["Risk_Tier"].isin(tiers)]

# Display dataset
st.subheader("Filtered Farmer Data")
st.dataframe(filtered_df)

# Plot distribution
st.subheader("Credit Score Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df["Credit_Score"], kde=True, bins=20, ax=ax)
st.pyplot(fig)

# Predict new entry
st.subheader("📊 Predict Credit Tier for New Farmer")
input_data = {}
input_data["Age"] = st.slider("Age", 18, 25, 22)
input_data["Farm_Size_HA"] = st.slider("Farm Size (ha)", 0.5, 5.0, 2.0)
input_data["Years_of_Experience"] = st.slider("Years of Experience", 1, 10, 3)
input_data["Average_Annual_Yield_Tons"] = st.slider("Yield (tons)", 1.0, 10.0, 5.0)
input_data["Income_per_Year_NGN"] = st.number_input("Annual Income (NGN)", 200000, 2000000, 500000)
input_data["Repayment_History_Score"] = st.slider("Repayment Score", 0.0, 1.0, 0.7)

# Encode categorical inputs
input_data["Gender"] = st.selectbox("Gender", ["Male", "Female"])
input_data["Education_Level"] = st.selectbox("Education", ["None", "Primary", "Secondary", "Tertiary"])
input_data["Previous_Loan"] = st.selectbox("Previous Loan?", ["Yes", "No"])
input_data["Mobile_Money_Use"] = st.selectbox("Mobile Money Use?", ["Yes", "No"])
input_data["Market_Access"] = st.selectbox("Market Access", ["Good", "Average", "Poor"])

# Manual encoding (must match training encoding)
def manual_encode(value, column):
    maps = {
        "Gender": {"Male": 1, "Female": 0},
        "Education_Level": {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3},
        "Previous_Loan": {"No": 0, "Yes": 1},
        "Mobile_Money_Use": {"No": 0, "Yes": 1},
        "Market_Access": {"Poor": 2, "Average": 1, "Good": 0}
    }
    return maps[column][value]

for col in ["Gender", "Education_Level", "Previous_Loan", "Mobile_Money_Use", "Market_Access"]:
    input_data[col] = manual_encode(input_data[col], col)

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

st.success(f"Predicted Credit Risk Tier: **{prediction}**")
