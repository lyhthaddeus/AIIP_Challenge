from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, root_mean_squared_error

def train_xgb_model(X_train, y_train, X_test, y_test, params=None):
    """
    Train and evaluate the XGBoost model.
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',  # Regression task
            'eval_metric': 'rmse',            # Evaluation metric
            'learning_rate': 0.1,             # Learning rate
            'max_depth': 6,                   # Max tree depth
            'n_estimators': 1000,              # Number of boosting rounds
        }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)


    # Predict and evaluate
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    evaluate_model(y_test, y_pred)
    
    return model

def evaluate_model(y_true, y_pred):
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()