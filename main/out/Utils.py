import pandas as pd 
import math 
import numpy as np
import datetime
import pickle as pk

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def pkload(file_path):
    file = open(file_path, 'rb')
    aaa = pk.load(file)
    file.close()
    return aaa

def read_file(time_interval:int,data_start_day:int,data_end_day:int,daystart,dayend):
# def read_file(time_interval:int):
#     global data_start_day,data_end_day,daystart,dayend
    filename_in = './data/data_'+str(time_interval)+'m_in.csv'
    filename_out = './data/data_'+str(time_interval)+'m_out.csv'

    file_in = pd.read_csv(filename_in)
    file_out = pd.read_csv(filename_out)

    file_in['startTime']= pd.to_datetime(file_in['startTime'])
    file_out['startTime']= pd.to_datetime(file_out['startTime'])

    file_in = file_in[file_in['startTime'].dt.day.isin(np.arange(data_start_day, data_end_day+1))]
    file_out = file_out[file_out['startTime'].dt.day.isin(np.arange(data_start_day, data_end_day+1))]

    dayendt = datetime.datetime.strptime(dayend, '%H:%M') 
    dayendt = (dayendt + datetime.timedelta(seconds=-1)).strftime('%H:%M') 
    data_in = file_in.set_index("startTime").between_time(daystart, dayendt)
    data_out = file_out.set_index("startTime").between_time(daystart, dayendt)

    return data_in,data_out

def datasetToSeq(dataset, input_step:int,pred_len:int):
    total_num, stations, dims = dataset.shape
    sample_num = total_num - input_step - pred_len +1
    seq = np.zeros((sample_num, input_step+pred_len, stations, dims))
    for i in range(sample_num):
        seq[i] = dataset[i: i+input_step+pred_len]
    return seq

def datasetToSeq_daybyday(dataset, input_step:int, pred_len:int, day_total_step:int, day:int):
    day_samples = day_total_step
    total_num, stations, dims = dataset.shape
    sample_num = day_samples - input_step - pred_len + 1
# seq_data
    seq_data_temp = []
    for i in range(day):
        seq_day= []
        temp = dataset[day_samples*i : day_samples*i+day_samples]
        for j in range(sample_num):
            seq_day.append(temp[j: j+pred_len+input_step])
        seq_day2 = np.array(seq_day)
        seq_data_temp.append(seq_day2)
    seq_data = np.concatenate(seq_data_temp, axis=0)
    return seq_data