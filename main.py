from src.data_loader import load_data, preprocess_data
from src.visualizer import plot_correlation_heatmap
from src.linear_model import run_linear_regression
from src.knn_model import run_knn_regression

ticker = 'GOOG'
start_date = '2010-01-01'
end_date = '2025-01-01'

print("📈 Loading data...")
data = load_data(ticker, start_date, end_date)

print("🔧 Preprocessing data...")
processed_data = preprocess_data(data)

print("📊 Plotting correlation heatmap...")
plot_correlation_heatmap(processed_data)

print("🏃‍♀️ Running Linear Regression...")
run_linear_regression(processed_data)

print("🏃‍♂️ Running KNN Regression...")
run_knn_regression(processed_data)
