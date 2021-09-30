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
from TCN_GCNDGL5 import *
from Param import *
import dgl


################# python input parameters #######################
parser = argparse.ArgumentParser()
parser.add_argument('--day',type=int,default=5,help='pred length')
parser.add_argument('--horizon',type=int,default=6,help='pred length')
parser.add_argument('cuda',type=int,default=3,help='adj data path')
args = parser.parse_args() #python
# args = parser.parse_args(args=[])    #jupyter notebook
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")
DAY = int(args.day)
CHANNEL = 3 + DAY
START_INDEX = DAY*105
################# Parameter Setting #######################
MODELNAME = 'Ablation_TCN_GCNDGL5_'+str(args.horizon)+'_day_'+str(args.day)
KEYWORD = MODELNAME + '_' + TASK +'_CHANNEL3_in+out+time_'  + DATANAME  + '_' + datetime.datetime.now().strftime("%y%m%d%H%M")
PATH = './save/' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
torch.backends.cudnn.deterministic = True
###########################################################
# read data
daystartt = datetime.datetime.strptime(DAYSTART, '%H:%M') 
dayendt = datetime.datetime.strptime(DAYEND, '%H:%M') 
day_minutes = int((dayendt - daystartt).total_seconds()/60)  #每天有多少分钟的数据
day_total_step = math.ceil(day_minutes/TIME_INTERVAL)   #对应每天有多少个step
data_in,data_out = read_file(TIME_INTERVAL,DATA_START_DAY,DATA_END_DAY,DAYSTART,DAYEND)  #data数据读取，并且截取 1号到 25号， 06:00 到 23:30的数据
if TASK is "in":
    data = data_in
else :
    data_in = data_in
    data_out = data_out
# Normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler,PowerTransformer,QuantileTransformer

scaler = MinMaxScaler()
# scaler = StandardScaler()
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
###############################################################
TRAIN_NUM = int(2608 * TRAINRATIO)
# adj_preprocess
# statistic adj  process
statistic_adjname = ['distance','traveltime','dtwall','od']
# static_gcn_adj
static_gcn_adj = []
for adjname_ in statistic_adjname:
    gcn_adj = single_adj_process(adjname_,ADJTYPE)
    static_gcn_adj.append(gcn_adj)
static_gcn_adj = np.array(static_gcn_adj)  # [S,K,N,N]  [4,2,N,N]
# static_gat_adj
static_gat_adj = []
for adjname_ in statistic_adjname:
    filename = './data/adj/W_'+str(adjname_)+'.csv'
    adj_mx = np.array(pd.read_csv(filename)).astype(np.float32)
    gat_adj = get_normalized_adj(adj_mx)
    static_gat_adj.append(gat_adj)
static_gat_adj = np.array(static_gat_adj)  # [S,N,N]  [4,N,N]

static_gcn_adj = torch.Tensor(static_gcn_adj).to(device)   # [S,K,N,N]  [4,2,N,N]
static_gat_adj = torch.Tensor(static_gat_adj)  # [S,N,N]  [4,N,N]
################################################
# dynamic_gat_adj
dynamic_gat_adj=[]
dynamic_adjname = ['od','dtw']
for dy_adjnale in dynamic_adjname:
    gat_filename = './data/adj/'+str(dy_adjnale)+'_adj_1012.pk'
    gatfile  = pkload(gat_filename)
    gat_values = list(gatfile.values())
    gat_values = np.array(gat_values)
    temp=[]
    for i in range(gat_values.shape[0]):
        adj = gat_values[i,:,:]
        norm_adj = get_normalized_adj(adj)
        temp.append(norm_adj)
    gat_norm = np.array(temp)
    dynamic_gat_adj.append(gat_norm)
dynamic_gat_adj = np.array(dynamic_gat_adj).transpose(1,0,2,3)  # [sample,D,N,N]
train_dynamic_gat = dynamic_gat_adj[START_INDEX:TRAIN_NUM]
test_dynamic_gat = dynamic_gat_adj[TRAIN_NUM:]
# dynamic_gcn_adj
dynamic_gcn_adj=[]
dynamic_adjname = ['od','dtw']
for dy_adjnale in dynamic_adjname:
    gcn_filename = './data/adj/'+str(dy_adjnale)+'_dynamic_gcn_graph_pk.pk'
    gcnfile  = pkload(gcn_filename)
    gcndata = gcnfile[ADJTYPE]
    dynamic_gcn_adj.append(gcndata)
dynamic_gcn_adj = np.array(dynamic_gcn_adj).transpose(1,0,2,3,4)  # [sample,D,K,N,N]
train_dynamic_gcn = dynamic_gcn_adj[START_INDEX:TRAIN_NUM]
print('START_INDEX ',START_INDEX)
print('TRAIN_NUM ',TRAIN_NUM)
print('train_dynamic_gcn shape',train_dynamic_gcn.shape)
test_dynamic_gcn = dynamic_gcn_adj[TRAIN_NUM:]    

train_dynamic_gat = torch.Tensor(train_dynamic_gat)
test_dynamic_gat = torch.Tensor(test_dynamic_gat)  # [522,2,81,81]
train_dynamic_gcn = torch.Tensor(train_dynamic_gcn).to(device)
test_dynamic_gcn = torch.Tensor(test_dynamic_gcn).to(device)   # [522,2,2,81,81]

###########################################################

def getXSYS(data, mode):
    TRAIN_NUM = int(2608 * TRAINRATIO)
    data_out = data[:,:,1:2]
    XS,YS = [],[]
    for i in range(START_INDEX,data.shape[0]-PRED_STEP-INPUT_STEP+1):
        x1 = data[i:i+INPUT_STEP]
        his_day_index  = [j for j in range(INPUT_STEP-105*DAY+i, INPUT_STEP+i,105*1)]
        temp = [data_out[k:k+INPUT_STEP] for k in his_day_index]
        HIS_Y = []
        for l in range(DAY):
            HIS_Y.append(temp[l])
        HIS = np.concatenate(HIS_Y, axis=-1)
        x = np.concatenate([x1,HIS], axis=2) # (12,81,DAY)
        y = data[i+INPUT_STEP:i+INPUT_STEP+PRED_STEP,:,1:2]
        XS.append(x)
        YS.append(y)
    XS = np.array(XS)
    YS = np.array(YS)
    print('XS shape: ',XS.shape)
    if mode =='TRAIN':
        XS = XS[0:TRAIN_NUM-START_INDEX].transpose(0,3,1,2)
        YS = YS[0:TRAIN_NUM-START_INDEX].transpose(0,3,1,2)
    if mode =='TEST':
        XS = XS[TRAIN_NUM-START_INDEX:].transpose(0,3,1,2)
        YS = YS[TRAIN_NUM-START_INDEX:].transpose(0,3,1,2)
    return XS,YS
# XS, YS shape is : samples * channel * (input_step or pred_step) * n_route   XS (522, 3, 12, 81)   YS (522, 1, 12, 81)
def getModel(name):
    global Lk
    ks, kt, bs_hour, bs_day, T, n, p = 3, 3, [[3, 16, 64], [64, 16, 64]], [[args.day, 16, 64], [64, 16, 64]], INPUT_STEP, N_NODE, 0
    model = TCN_DGLGCN(ks, kt, bs_hour, bs_day, T, n, p, device, N_adj=[len(statistic_adjname),len(dynamic_adjname)], gcn_layers=1, gat_layers=1).to(device)
#     model = TCN_DGLGCN(ks, kt, bs, T, n, p, device, N_adj=[len(statistic_adjname),len(dynamic_adjname)], gcn_layers=1, gat_layers=1).to(device)
#     model = STGCN_DGLGCN(ks, kt, bs, T, n, Lk, g2, p, device).to(device)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y, dynamic_gat_adj, dynamic_gcn_adj in data_iter:
            dynamic_gat_adj = dynamic_gat_adj.transpose(1,0)
            dynamic_gcn_adj = dynamic_gcn_adj.transpose(1,0)            
            y_pred = model(x,static_gat_adj,static_gcn_adj,dynamic_gat_adj,dynamic_gcn_adj)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y, dynamic_gat_adj, dynamic_gcn_adj in data_iter:
            dynamic_gat_adj = dynamic_gat_adj.transpose(1,0)
            dynamic_gcn_adj = dynamic_gcn_adj.transpose(1,0)            
            YS_pred_batch = model(x,static_gat_adj,static_gcn_adj,dynamic_gat_adj,dynamic_gcn_adj)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', INPUT_STEP, PRED_STEP)
    model = getModel(name)
#     summary(model, (CHANNEL, INPUT_STEP, N_NODE), device=device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch, train_dynamic_gat, train_dynamic_gcn)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    print('LOSS is :',LOSS)
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    min_val_loss = np.inf
    wait = 0
    for epoch in range(200):
        starttime = datetime.datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y, dynamic_gat_adj, dynamic_gcn_adj in train_iter:
            optimizer.zero_grad()
            dynamic_gat_adj = dynamic_gat_adj.transpose(1,0)
            dynamic_gcn_adj = dynamic_gcn_adj.transpose(1,0)
            y_pred = model(x,static_gat_adj,static_gcn_adj,dynamic_gat_adj,dynamic_gcn_adj)
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
        print("epoch", epoch+1, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.3f, %s, %.3f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.3f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.3f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())
        
def testModel(name, mode, XS, YS):
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', INPUT_STEP, PRED_STEP)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch, test_dynamic_gat, test_dynamic_gcn)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    print('LOSS is :',LOSS)
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred)
    YS = scaler.inverse_transform(YS)
    YS_pred = scaler.inverse_transform(YS_pred)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.3f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.3f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Testing Ended ...', time.ctime())


def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('TCN_GCNDGL5.py', PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('pred_TCN_GCNDGL5.py', PATH)

    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(data3, 'TRAIN')
    if args.horizon != 0:
        trainYS = trainYS[:,:,args.horizon-1:args.horizon,:]
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS)
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data3, 'TEST')
    if args.horizon != 0:
        testYS = testYS[:,:,args.horizon-1:args.horizon,:]   
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'test', testXS, testYS)

    
if __name__ == '__main__':
    main()

