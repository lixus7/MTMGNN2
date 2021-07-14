import pandas as pd
import scipy.sparse as ss
import csv
import numpy as np
import os
import shutil
import sys
import time
import datetime
import Metrics
import math
from Utils import *
from Param import *

HISTORYDAY = 7
DAYTIMESTEP = 210

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':
        # Be careful here, start with 7*DAYTTIMESTEP for this method.
        for i in range(7*DAYTIMESTEP, TRAIN_NUM - PRED_STEP - INPUT_STEP + 1):
            x = data[i:i+INPUT_STEP, :]
            y = data[i+INPUT_STEP:i+INPUT_STEP+PRED_STEP, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - INPUT_STEP,  data.shape[0] - PRED_STEP - INPUT_STEP + 1):
            x = data[i:i+INPUT_STEP, :]
            y = data[i+INPUT_STEP:i+INPUT_STEP+PRED_STEP, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

def getXSWeek(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    YS, YS_index = [], []
    if mode == 'TRAIN':  
        for i in range(HISTORYDAY*DAYTIMESTEP, TRAIN_NUM - PRED_STEP - INPUT_STEP + 1):
            YS_index.append(np.arange(i+INPUT_STEP, i+INPUT_STEP+PRED_STEP))
            YS.append(data[i+INPUT_STEP:i+INPUT_STEP+PRED_STEP, :])
        YS_index, YS = np.array(YS_index), np.array(YS)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - INPUT_STEP,  data.shape[0] - PRED_STEP - INPUT_STEP + 1):
            YS_index.append(np.arange(i+INPUT_STEP, i+INPUT_STEP+PRED_STEP))
            YS.append(data[i+INPUT_STEP:i+INPUT_STEP+PRED_STEP, :])
        YS_index, YS = np.array(YS_index), np.array(YS)
    
    XS_Week = []
    for i in range(YS_index.shape[0]):
        Week = []
        for j in range(YS_index.shape[1]):
            index = YS_index[i, j]
            Week.append(data[index-HISTORYDAY*DAYTIMESTEP:index:DAYTIMESTEP, :])
        XS_Week.append(Week)
    XS_Week = np.array(XS_Week)
    return XS_Week, YS

def HistoricalAverage(data, mode):
    XS_Week, YS = getXSWeek(data, mode)
    YS_pred = np.mean(XS_Week, axis=2)
    return YS_pred

def testModel(name, data, mode):
    print('INPUT_STEP, PRED_STEP', INPUT_STEP, PRED_STEP)
    XS, YS = getXSYS(data, mode)
    YS_pred = HistoricalAverage(data, mode)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(PRED_STEP):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()

################# Parameter Setting #######################
MODELNAME = 'HistoricalAverage'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.datetime.now().strftime("%y%m%d%H%M")
PATH = './save'
# PATH = './save/' + KEYWORD
################# Parameter Setting #######################
# read data
daystartt = datetime.datetime.strptime(DAYSTART, '%H:%M') 
dayendt = datetime.datetime.strptime(DAYEND, '%H:%M') 
day_minutes = int((dayendt - daystartt).total_seconds()/60)  #每天有多少分钟的数据
day_total_step = math.ceil(day_minutes/TIME_INTERVAL)   #对应每天有多少个step
data_in,data_out = read_file(TIME_INTERVAL,DATA_START_DAY,DATA_END_DAY,DAYSTART,DAYEND)  #data数据读取，并且截取 1号到 25号， 06:00 到 23:30的数据
if TASK == "in":
    data = data_in
else :
    data = data_out
data=data.values
###########################################################
def main():
#     if not os.path.exists(PATH):
#         os.makedirs(PATH)
    currentPython = sys.argv[0]

    print(KEYWORD, 'testing started', time.ctime())
    testModel(MODELNAME, data, 'TEST')


if __name__ == '__main__':
    main()