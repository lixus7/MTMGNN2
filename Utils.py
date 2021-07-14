import pandas as pd 
import math 
import numpy as np
import datetime

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
    seq = np.zeros((sample_num, input_step+pred_len, stations, dims))
# seq_data
    for i in range(day):
        temp = dataset[day_samples*i : day_samples*i+day_samples]
        for j in range(sample_num):
            seq[j] = temp[j: j+pred_len+input_step]
        if i==0:
            seq_data = seq
        else:
            seq_data = np.concatenate((seq_data, seq), axis=0)
    return seq_data