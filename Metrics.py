import numpy as np
import pandas as pd

def MSE(y_true, y_pred):
    '''
     R Mean squared error
    '''
    mse = np.square(y_pred - y_true)
    mse = np.mean(mse)
    return mse

def RMSE(y_true, y_pred):
    '''
     R Mean squared error
    '''
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MAE(y_true, y_pred):
    '''
    Mean absolute error
    '''
    return np.mean(np.abs(y_true - y_pred))

def MAPE(y_true, y_pred, null_val=0):
    '''
    MAPE
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100    
def evaluate(y_true,y_pred):
    y_true[y_true<1e-1]=0
    y_pred[y_pred<1e-1]=0
    mse = MSE(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    mae = MAE(y_true, y_pred)
    mape = MAPE(y_true, y_pred, null_val=0)
    return mse,rmse,mae,mape