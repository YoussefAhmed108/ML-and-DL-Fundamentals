import numpy as np

def r2_score(y_true, y_pred):
    # Calculate the mean of the true values
    y_true_mean = np.mean(y_true)
    # Calculate the total sum of squares
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    # Calculate the residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    # Calculate the R^2 score
    r2 = 1 - (ss_res / ss_tot)
    return r2

def mean_squared_error(y_true, y_pred):
    # Calculate the mean squared error
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def mean_absolute_error(y_true, y_pred):
    # Calculate the mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def root_mean_squared_error(y_true, y_pred):
    # Calculate the root mean squared error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse