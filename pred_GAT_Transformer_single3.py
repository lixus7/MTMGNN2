import sys
import argparse
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import Metrics
from Utils import *
from GAT_Transformer import *
from Param import *
# from Param_GraphWaveNet1 import *

CHANNEL = 3

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    train_set = data[0:TRAIN_NUM]
    test_set = data[TRAIN_NUM:]
    XS, YS = [], []
    if mode == 'TRAIN': 
        XS = train_set[:,:,0:INPUT_STEP,:]
        YS = train_set[:,:,INPUT_STEP:INPUT_STEP+PRED_STEP,:]
    elif mode == 'TEST':
        XS = test_set[:,:,0:INPUT_STEP,:]
        YS = test_set[:,:,INPUT_STEP:INPUT_STEP+PRED_STEP,:]
    XS = XS.transpose(0, 1, 3, 2)
    YS = YS.transpose(0, 3, 2, 1)
    YS = YS[:,:,:,1]
    return XS, YS
# XS, YS shape is : samples * channel * (input_step or pred_step) * n_route    TEST XS (1046, 2, 81, 12)  TEST YS (1046, 81, 12)


def getModel(name):
    model = GAT_Transformer(in_channel = CHANNEL, embed_size = 64, time_num = 82 , num_layers = 3, T_dim = INPUT_STEP, output_T_dim = 1, heads = 8, forward_expansion = 4, gpu = device, dropout = 0).to(device)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x,W)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x,W)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', INPUT_STEP, PRED_STEP)
    model = getModel(name)
    summary(model, [(CHANNEL, N_NODE, INPUT_STEP),(N_NODE,N_NODE)], device=device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True, drop_last=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    print('LOSS is :',LOSS)
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    min_val_loss = np.inf
    wait = 0
    for epoch in range(EPOCH):
        starttime = datetime.datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x,W)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.3f, %s, %.3f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS = YS.transpose(0,2,1)
    YS_pred = YS_pred.transpose(0,2,1)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.3f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.3f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())
        
def testModel(name, mode, XS, YS):
    if LOSS == "MaskMAE":
        criterion = Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    print('Model Testing Started ...', time.ctime())
    print('INPUT_STEP, PRED_STEP', INPUT_STEP, PRED_STEP)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH+ '/' + name + '.pt'))
    
    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS = YS.transpose(0,2,1)
    YS_pred = YS_pred.transpose(0,2,1)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.3f" % (name, mode, torch_score, torch_score))
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.3f\n" % (name, mode, torch_score, torch_score))
    print("horizon %s : , %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (str(args.horizon),name, mode, MSE, RMSE, MAE, MAPE))
    f.write("horizon %s : , %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (str(args.horizon),name, mode, MSE, RMSE, MAE, MAPE))
#     for i in range(PRED_STEP):
#         MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
#         print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
#         f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('Model Testing Ended ...', time.ctime())
################# python input parameters #######################
parser = argparse.ArgumentParser()
parser.add_argument('--adjdata',type=str,default='W_distance.csv',help='adj data path')
parser.add_argument('--horizon',type=int,default=1,help='pred length')
parser.add_argument('cuda',type=int,default=3,help='cuda device number')  
args = parser.parse_args() #python    
# args = parser.parse_args(args=[])    #jupyter notebook
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")
################# Parameter Setting #######################
MODELNAME = 'Transformersingle3_horizon_'+str(args.horizon)
KEYWORD = MODELNAME + '_' + TASK  +' _CHANNEL3_in+out+time_ADJ'  + str(args.adjdata) + datetime.datetime.now().strftime("%y%m%d%H%M")
PATH = './save/' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100) 
np.random.seed(100)
torch.backends.cudnn.deterministic = True
###########################################################
# read data
adjpath = './data/adj/'+args.adjdata
A = pd.read_csv(adjpath).values
if np.max(A)>100:
    adj_mx = pd.read_csv(pkl_filename).values
    distances = adj_mx[~np.isinf(adj_mx)].flatten()
    std = distances.std()
    W = np.exp(-np.square(adj_mx / std))  
#         W = weight_matrix(A)
else:
    W = A
W = torch.Tensor(W).to(device)    

daystartt = datetime.datetime.strptime(DAYSTART, '%H:%M') 
dayendt = datetime.datetime.strptime(DAYEND, '%H:%M') 
day_minutes = int((dayendt - daystartt).total_seconds()/60)  #每天有多少分钟的数据
day_total_step = math.ceil(day_minutes/TIME_INTERVAL)   #对应每天有多少个step
data_in,data_out = read_file(TIME_INTERVAL,DATA_START_DAY,DATA_END_DAY,DAYSTART,DAYEND)  #data数据读取，并且截取 1号到 25号， 06:00 到 23:30的数据
if TASK == "in":
    data = data_in
else :
    data_in = data_in
    data_out = data_out
# Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_in_nor = scaler.fit_transform(data_in)
data_out_nor = scaler.fit_transform(data_out)
data_in_nor = data_in_nor.reshape(data_in_nor.shape[0],data_in_nor.shape[1],1)
data_out_nor = data_out_nor.reshape(data_out_nor.shape[0],data_out_nor.shape[1],1)
data_nor = np.concatenate((data_in_nor, data_out_nor), axis = 2)
# add time channel
num_samples, num_nodes = data_in.shape
time_ind = (data_in.index.values - data_in.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
print('time_ind.shape:',time_ind.shape)
time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
print('time_in_day.shape:',time_in_day.shape)
data3 = np.concatenate((data_nor,time_in_day), axis=2)
#data 转换成sequence
# seq_data = datasetToSeq_daybyday(data_nor, INPUT_STEP,PRED_STEP, day_total_step, day=DATA_END_DAY-DATA_START_DAY+1)
seq_data = datasetToSeq(data3,INPUT_STEP,PRED_STEP)
seq_data = seq_data.transpose(0,3,1,2)
# seq_data shape is : samples * channel * (input_step+pred_step) * n_route   (5227, 3, 24, 81)
print('seq_data shape: ',seq_data.shape)
###########################################################
def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('GAT_Transformer.py', PATH)
    shutil.copy2('pred_GAT_Transformer_single3.py', PATH)
    shutil.copy2('Param.py', PATH)
    
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(seq_data, 'TRAIN')
    if args.horizon != 0:
        trainYS = trainYS[:,:,args.horizon-1:args.horizon]
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS)
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(seq_data, 'TEST')
    if args.horizon != 0:
        testYS = testYS[:,:,args.horizon-1:args.horizon]    
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'test', testXS, testYS)

    
if __name__ == '__main__':
    main()

