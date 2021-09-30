import numpy as np
import pandas as pd

# def MSE(y_true, y_pred):
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype(np.float32)
#         mask /= np.mean(mask)
#         mse = np.square(y_pred - y_true)
#         mse = np.nan_to_num(mse * mask)
#         mse = np.mean(mse)
#         return mse
    
# def RMSE(y_true, y_pred):
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype(np.float32)
#         mask /= np.mean(mask)
#         rmse = np.square(np.abs(y_pred - y_true))
#         rmse = np.nan_to_num(rmse * mask)
#         rmse = np.sqrt(np.mean(rmse))
#         return rmse
        
# def MAE(y_true, y_pred):
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype(np.float32)
#         mask /= np.mean(mask)
#         mae = np.abs(y_pred - y_true)
#         mae = np.nan_to_num(mae * mask)
#         mae = np.mean(mae)
#         return mae

# def MAPE(y_true, y_pred, null_val=0):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         if np.isnan(null_val):
#             mask = ~np.isnan(y_true)
#         else:
#             mask = np.not_equal(y_true, null_val)
#         mask = mask.astype('float32')
#         mask /= np.mean(mask)
#         mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
#         mape = np.nan_to_num(mask * mape)
#         return np.mean(mape) * 100

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
    y_true[y_true<0.5]=0
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
# def weighted_mean_absolute_percentage_error(Y_true, Y_pred):
# 	# (n * 276)
# 	total_sum=np.sum(Y_true)
# 	average=[]
# 	for i in range(len(Y_true)):
# 		for j in range(len(Y_true[0])):
# 			if Y_true[i][j]>0:
# 				temp=(Y_true[i][j]/total_sum)*np.abs((Y_true[i][j] - Y_pred[i][j]) / Y_true[i][j])
# 				average.append(temp)
# 	return np.sum(average)  

def weighted_mean_absolute_percentage_error(Y_true, Y_pred):
    Y_true = Y_true.reshape(Y_true.shape[0],-1)
    Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)
    total_sum=np.sum(Y_true)
    average=[]
    for i in range(Y_true.shape[0]):
        for j in range(Y_true.shape[1]):
            if Y_true[i,j]>0:
                temp = (Y_true[i,j]/total_sum)*np.abs((Y_true[i,j]-Y_pred[i,j])/(Y_true[i,j]+0.1))
                average.append(temp)
    return np.sum(average)*100
# def weighted_mean_absolute_percentage_error(Y_true, Y_pred):
    
# # (n * 276)
#     total_sum=np.sum(Y_true)
#     temp = np.sum(np.abs(Y_true-Y_pred))
#     wmape = total_sum/temp
#     return wmape    
    
def evaluate(y_true,y_pred):
    y_true[y_true<1]=0
    mse = MSE(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    mae = MAE(y_true, y_pred)
    wmape = weighted_mean_absolute_percentage_error(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    return mse,rmse,mae,wmape