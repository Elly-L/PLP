# app.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset (same as your project)
@st.cache_data  # Cache the data to speed up the app
def load_data():
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    data = pd.read_csv(url)
    data_2020 = data[data['year'] == 2020].dropna(subset=['co2', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2'])
    return data_2020

data = load_data()

# Train the model (same as your project)
X = data[['co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2']]
y = data['co2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# App title and description
st.title("🌍 CO₂ Emissions Predictor for Climate Action by Elly Odhiambo")
st.markdown("""
This app predicts a country's **total CO₂ emissions** based on its energy consumption data.
**Goal:** Help policymakers identify high-emission regions for targeted climate action (SDG 13).
""")

# User inputs for prediction
st.header("Predict CO₂ Emissions")
co2_per_capita = st.number_input("CO₂ per Capita (metric tons)", min_value=0.0, value=5.0)
coal_co2 = st.number_input("CO₂ from Coal (thousand metric tons)", min_value=0.0, value=1000.0)
oil_co2 = st.number_input("CO₂ from Oil (thousand metric tons)", min_value=0.0, value=2000.0)
gas_co2 = st.number_input("CO₂ from Gas (thousand metric tons)", min_value=0.0, value=1000.0)

# Predict button
if st.button("Predict Emissions"):
    input_data = pd.DataFrame({
        'co2_per_capita': [co2_per_capita],
        'coal_co2': [coal_co2],
        'oil_co2': [oil_co2],
        'gas_co2': [gas_co2]
    })
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted CO₂ Emissions: **{prediction:,.2f} thousand metric tons**")

# Show dataset preview
st.header("Dataset Preview (2020)")
st.dataframe(data[['country', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'co2']].head(10))

# Model performance
st.header("Model Performance")
st.write(f"- **Mean Absolute Error (MAE):** {23.39} thousand metric tons")
st.write("- The model was trained on 101 samples and tested on 26 samples.")
st.image("https://mistralaichatupprodswe.blob.core.windows.net/chat-images/7b/28/ec/7b28ec4e-fbd0-4144-97d8-42d67cf2037f/a470b4fa-2711-4026-ac67-7ea6350d9c30/d19dc31c-3304-4aa4-918b-db079c7e935a", caption="Actual vs Predicted CO₂ Emissions (2020)", width=500)
