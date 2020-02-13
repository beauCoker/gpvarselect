import numpy as np
from scipy.stats import multivariate_normal

def standardize_data(original_x_train, original_y_train, original_x_test, original_y_test,
                     mean_x = None, std_x = None, mean_y = None, std_y = None):
    
    if mean_x is not None and std_x is not None:
        mean_x, std_x = np.mean(original_x_train), np.std(original_x_train)

    if mean_y is not None and std_y is not None:
        mean_y, std_y = np.mean(original_y_train), np.std(original_y_train)
    
    train_x = (original_x_train - mean_x) / std_x
    train_y = ((original_y_train - mean_y) / std_y).reshape(-1,1)

    test_x = ((original_x_test - mean_x) / std_x).reshape(-1,1)
    test_y = ((original_y_test - mean_y) / std_y).reshape(-1,1)

    return train_x, train_y, test_x, test_y


## posterior metrics

def rmse(f, f_pred):
    # predictive root mean squared error (RMSE)
    return np.sqrt(np.mean((f - f_pred)**2))

def picp(f, f_pred_lb, f_pred_ub):
    # prediction interval coverage (PICP)
    return np.mean(np.logical_and(f >= f_pred_lb, f <= f_pred_ub))

def mpiw(f_pred_lb, f_pred_ub):
    # mean prediction interval width (MPIW)
    return np.mean(f_pred_ub - f_pred_lb)

def test_log_likelihood(mean, cov, test_y):
    return multivariate_normal.logpdf(test_y.reshape(-1), mean.reshape(-1), cov)

