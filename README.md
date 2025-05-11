
# 📈 Google Stock Price Prediction using Machine Learning

This project implements machine learning techniques to predict Google (GOOG) stock prices using historical data fetched via the `yfinance` API. The goal is to model and evaluate stock closing prices based on various market features like Open, High, Low, and Volume using both **Linear Regression** and **K-Nearest Neighbors (KNN) Regression**.

---

## 🗃️ Dataset

- **Source:** [Yahoo Finance via yfinance](https://pypi.org/project/yfinance/)
- **Ticker:** GOOG
- **Date Range:** 2010-01-01 to 2025-01-01
- **Features Used:** Open, High, Low, Volume
- **Target:** Close (Closing Price)

---

## 🔧 Requirements

Install the required packages with:

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn
```

---

## 📊 Project Workflow

### 1. **Data Collection**
- Historical stock prices fetched using `yfinance`.

### 2. **Data Preprocessing**
- Checked for null values and duplicates.
- Visualized data distribution using Seaborn's pairplot and correlation heatmaps.

### 3. **Exploratory Data Analysis**
- Summary statistics generated with `describe()`.
- Heatmap used to show correlations between features.

### 4. **Feature Engineering**
- Features: `High`, `Low`, `Open`, `Volume`
- Target: `Close`

### 5. **Model Training & Evaluation**

#### 📌 Linear Regression
- Train/Test split using `train_test_split()`
- Achieved **R² ≈ 0.9999** on both train and test data.
- Evaluated using:
  - R² Score
  - MAE, MSE, RMSE
- Visualized with:
  - Line plot (Actual vs Predicted)
  - Scatter plot

#### 📌 KNN Regression
- Features scaled using `MinMaxScaler`.
- Value of `k=5`
- Achieved high accuracy with:
  - Test R² ≈ 0.9998
  - RMSE ≈ 0.68
- Rolling prediction technique implemented for time-series forecasting.

---

## 📈 Visualizations
- Pairplots for feature relationships
- Correlation heatmaps
- Line and bar plots for predicted vs actual values
- Final time-series forecast plot for Google stock prices

---

