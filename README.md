# Wind Power Prediction Application

## Overview
This project presents an **AI-based Wind Power Prediction System** developed using Long Short-Term Memory (LSTM) networks. The goal is to accurately forecast wind turbine power output by analyzing key environmental and historical parameters. The system leverages time-series modeling to capture temporal dependencies and improve prediction performance. A user-friendly web application is built using Streamlit to enable real-time predictions and interactive analysis. The system aims to improve renewable energy planning, grid stability, and decision-making through data-driven insights.

---

## Features
- **LSTM-Based Forecasting**

  - Implemented LSTM model for time-series prediction
  - Captures temporal dependencies in wind turbine data
  - Achieved **R² ≈ 0.92**, indicating high accuracy
  
- **Time-Series Modeling**
  
  - Incorporated lag features (Lag_1, Lag_2)
  - Modeled daily and seasonal variations using Hour and Month
  
- **Feature Engineering**

  - Processed SCADA dataset with:
       - Wind Speed
       - Wind Direction
       - Theoretical Power Curve
       - LV Active Power (target)
    Applied:
       - Data cleaning and outlier removal
       - Normalization using MinMaxScaler
         
- **Model Evaluation**

  - Evaluated using standard metrics:
     - MAE (Mean Absolute Error)
     - MSE (Mean Squared Error)
     - RMSE (Root Mean Squared Error)
     - R² Score
       
- **Streamlit Web Application**

  - Interactive UI for user inputs
  - Real-time wind power prediction
  - **Visualization of**:
  - Actual vs Predicted Power
  - Predicted Power vs Wind Speed

---
  
## Technology Stack

 - Programming Language: Python
 - Deep Learning: TensorFlow, Keras (LSTM)
 - Data Processing: Pandas, NumPy
 - Visualization: Matplotlib
 - Web Framework: Streamlit
 - Dataset: SCADA wind turbine data (CSV format)
   
---
   
## Results
 - The LSTM model demonstrated strong predictive performance for wind power forecasting. It achieved an R² score of    approximately 0.92, indicating high accuracy in capturing temporal patterns from the data.

- **Error Metrics**:

  - MAE: ~ 0.12
  - MSE: ~ 0.04
  - RMSE: ~ 0.20
  
---
  
## How To Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/wind-power-pridiction-using-AI.git
   cd wind-power-prediction-using-AI
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the Application**:
   Open the URL displayed in your terminal (e.g., http://localhost:8501) in a web browser.

---

## Dataset
 The dataset contains key variables for wind power Prediction, such as:
   - Wind Speed (m/s)
   - Wind Direction (°)
   - Theoretical Power Curve (KWh)
   - LV Active Power (kW)
   - Hour, Month
   - Lag_1, Lag_2
- The dataset was preprocessed to handle missing values, outliers.
  
---

## Future Work
 - Integration of IoT sensors for real-time data
 - Development of mobile/web dashboard
 - Use of advanced models (Transformer, CNN-LSTM)
 - Inclusion of additional weather parameters
 - Expansion to multiple wind farm locations

---
