import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import math


def run_linear_regression(data):
    x = data[['High', 'Low', 'Open', 'Volume']].values
    y = data[['Close']].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("Linear Regression Metrics:")
    print("R2 Score:", r2_score(y_test, predictions))
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", math.sqrt(mean_squared_error(y_test, predictions)))

    # Plot
    dframe = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})
    dframe.head(10).plot(kind='bar')
    plt.title('Actual vs Predicted (Linear Regression)')
    plt.ylabel('Closing Price')
    plt.show()
