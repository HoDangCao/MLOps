import numpy as np
import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate new incoming data
np.random.seed(42)
X = np.random.rand(200, 1) * 10
y = 3 * X.squeeze() + np.random.randn(200) * 2

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
mlflow.set_experiment("mlops_retraining")
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Log model and metrics
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

print(f"Model retrained & logged with MSE: {mse}")