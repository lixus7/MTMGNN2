import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from chinese_calendar import is_workday, is_holiday
import pickle as pk
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pickle as pk

def pksave(data, file_path):
    file = open(file_path, 'wb')
    pk.dump(data, file)
    file.close()
    return 

data_10m_in = pd.read_csv('../data_10m_in.csv')
data_10m_in['startTime']= pd.to_datetime(data_10m_in['startTime'])
data_10m_in = data_10m_in[data_10m_in['startTime'].dt.day<26]
data_10m_in = data_10m_in.set_index("startTime").between_time("06:00", "23:20")
time10 = data_10m_in[0:-17].index

def generete_one(data):
    num_sensors = 81
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = 0
    for i in range(81):
        x_id = i
        x = data.iloc[:,i]
        for j in range(81):
            y_id = j
            y = data.iloc[:,j]
            distance, path = fastdtw(x, y, dist=euclidean)
            dist_mx[x_id][y_id]=distance
    return dist_mx

def generate_odadj():
    freq = 600*12
    adj = {}
    starttime = datetime.datetime.now()
    for i in range(data_10m_in[0:-17].shape[0]):  
        time1 = datetime.datetime.now()
        time_start = str(time10[i])
        time_end = str(time10[i] + pd.Timedelta(seconds=freq))
        temp = data_10m_in[(data_10m_in.index<time_end)&(data_10m_in.index>=time_start)]
        dtw_one = generete_one(temp)
        adj[time10[i]]=dtw_one
        time2 = datetime.datetime.now()
        deltatime = (time2 - time1).seconds
        print('one adj12*10 time uses: ', deltatime, ' seconds')
    return adj
    endtime = datetime.datetime.now()
    alltime = (endtime - starttime).seconds
    print('alltime: ',alltime)
    hour = int(alltime/3600)
    minute = int((alltime-hour*3600)/60)
    second = alltime-hour*3600-minute*60
    print( "time used: ", hour, ' hours ',minute,' minutes', second,' second') 
    
def main():
    dtw_adj10 = generate_odadj()
    pksave(dtw_adj10, './dtw_adj_10x12.pk')   
    print('successed saved')
    
if __name__ == '__main__':
    main()