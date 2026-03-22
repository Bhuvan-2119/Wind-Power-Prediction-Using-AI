import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Folder path of current app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "WIND.csv")
    wind = pd.read_csv(csv_path)

    wind['Date/Time'] = pd.to_datetime(wind['Date/Time'], format="%d %m %Y %H:%M")
    wind['Date'] = wind['Date/Time'].dt.normalize()
    wind['Hour'] = wind['Date/Time'].dt.hour
    wind.drop(columns=['Date/Time'], axis=1, inplace=True)

    hourly_avg_power = wind.groupby(['Date', 'Hour'])[
        ['LV ActivePower (kW)', 'Theoretical_Power_Curve (KWh)',
         'Wind Speed (m/s)', 'Wind Direction (°)']
    ].mean().reset_index()

    hourly_avg_power['Month'] = hourly_avg_power['Date'].dt.month
    hourly_avg_power['Lag_1'] = hourly_avg_power['LV ActivePower (kW)'].shift(1)
    hourly_avg_power['Lag_2'] = hourly_avg_power['LV ActivePower (kW)'].shift(2)

    hourly_avg_power = hourly_avg_power.dropna()
    hourly_avg_power.set_index('Date', inplace=True)

    return hourly_avg_power

# Load dataset
data = load_data()

X = data.drop(['LV ActivePower (kW)'], axis=1)
y = data['LV ActivePower (kW)']

# Train-test split (80/20)
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Streamlit UI
st.title("Wind Power Prediction using AI")
st.markdown("<br>", unsafe_allow_html=True)

st.sidebar.header("Input Features")

hour = st.sidebar.slider("Hour", 0, 23, 12)
theoretical_power_curve = st.sidebar.number_input("Theoretical Power Curve (KWh)", min_value=0.0)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0)
wind_direction = st.sidebar.number_input("Wind Direction (°)", min_value=0.0)
month = st.sidebar.slider("Month", 1, 12, 1)
lag_1 = st.sidebar.number_input("Lag_1", min_value=0.0)
lag_2 = st.sidebar.number_input("Lag_2", min_value=0.0)

input_data = pd.DataFrame({
    'Hour': [hour],
    'Theoretical_Power_Curve (KWh)': [theoretical_power_curve],
    'Wind Speed (m/s)': [wind_speed],
    'Wind Direction (°)': [wind_direction],
    'Month': [month],
    'Lag_1': [lag_1],
    'Lag_2': [lag_2]
})

# ------------------- LSTM MODEL -------------------
st.subheader("🔹 LSTM Model Results")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit scalers on train data
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Reshape for LSTM: (samples, time_steps, features)
X_train_scaled = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_test_scaled = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])

# Load trained model
model_path = os.path.join(BASE_DIR, "best_lstm_model_0.92.h5")
model = load_model(model_path)

# Evaluate model on test data
pred_scaled = model.predict(X_test_scaled)
predicted = scaler_y.inverse_transform(pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# Clamp all predictions to zero or above
predicted = np.maximum(predicted, 0)

mae = mean_absolute_error(y_test_actual, predicted)
mse = mean_squared_error(y_test_actual, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, predicted)

# ---------------- PREDICTION ----------------
input_scaled = scaler_X.transform(input_data).reshape(1, 1, -1)
prediction = model.predict(input_scaled)
prediction = scaler_y.inverse_transform(prediction)[0][0]

# Clamp prediction to zero or above
prediction = max(prediction, 0)

st.success(f"⚡ Predicted Wind Power: **{prediction:.2f} kW**")
st.write("### 🔹 Model Performance")
st.write(f"MAE (Mean Absolute Error):  {mae:.2f}")
st.write(f"MSE (Mean Squared Error):  {mse:.2f}")
st.write(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
st.write(f"R² Score (Coefficient of Determination):  {r2:.2f}")

# ----------- Improved Graphs Button Section --------------------

button_html = """
    <style>
        .big-button {
            background-color: #1976d2;
            color: white;
            padding: 1em 2.5em;
            font-size: 1.2em;
            border: none;
            border-radius: 5px;
            margin: 1em 0 1em 0;
            cursor: pointer;
        }
        .big-button:hover {
            background-color: #125299;
            color: #f1f1f1;
        }
    </style>
    <center>
        <form action="" method="get">
            <button name="show_graphs" value="true" class="big-button" type="submit">
                Graphs
            </button>
        </form>
    </center>
"""
st.markdown(button_html, unsafe_allow_html=True)

show_graphs = st.query_params.get("show_graphs") == "true"

if show_graphs:
    st.write("## Data Visualization Graphs")

    # 1. Predicted Power vs Wind Speed (Appears First)
    st.write("### Predicted Power vs Wind Speed")
    wind_speeds = np.linspace(4, 14, 20)
    inputs = input_data.copy()
    outputs = []

    for ws in wind_speeds:
        inputs['Wind Speed (m/s)'] = ws
        input_scaled = scaler_X.transform(inputs).reshape(1, 1, -1)
        pred = model.predict(input_scaled)
        pred = max(scaler_y.inverse_transform(pred)[0][0], 0)
        outputs.append(pred)

    df_curve = pd.DataFrame({"Wind Speed (m/s)": wind_speeds, "Predicted Power (kW)": outputs})
    st.line_chart(df_curve.set_index("Wind Speed (m/s)"))

    # 2. Actual vs Predicted Wind Power (Smaller Graph)
    st.write("### Actual vs Predicted Wind Power (Sample Data)")
    N = 100  # Number of latest samples to plot
    fig, ax = plt.subplots(figsize=(6, 3.5))  # Medium-small size
    ax.plot(y_test_actual[-N:], label='Actual', marker='o', markersize=3, linewidth=1)
    ax.plot(predicted[-N:], label='Predicted', marker='x', markersize=3, linewidth=1)
    ax.set_title("Actual vs Predicted Wind Power")
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Sample")
    ax.legend()
    st.pyplot(fig)
