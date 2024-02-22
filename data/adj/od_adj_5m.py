from datetime import datetime
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from chinese_calendar import is_workday, is_holiday
#读取picke文件
import pickle as pk

def stationID(x):
    new_dict = {v : k for k, v in station_dict.items()}
    return new_dict[x]
station_dict = {
    0 : '湘湖',
    1 : '滨康路',
    2 : '西兴',
    3 : '滨和路',
    4 : '江陵路',
    5 : '近江',
    6 : '婺江路',
    7 : '城站',
    8 : '定安路',
    9 : '龙翔桥',
    10 : '凤起路',
    11 : '武林广场',
    12 : '西湖文化广场',
    13 : '打铁关',
    14 : '闸弄口',
    15 : '火车东站',
    16 : '彭埠',
    17 : '七堡',
    18 : '九和路',
    19 : '九堡',
    20 : '客运中心',
    21 : '下沙西',
    22 : '金沙湖',
    23 : '高沙路',
    24 : '文泽路',
    25 : '文海南路',
    26 : '云水',
    27 : '下沙江滨',
    28 : '乔司南',
    29 : '乔司',
    30 : '翁梅',
    31 : '余杭高铁站',
    32 : '南苑',
    33 : '临平',
    34 : '朝阳',
    35 : '曹家桥',
    36 : '潘水',
    37 : '人民路',
    38 : '杭发厂',
    39 : '人民广场',
    40 : '建设一路',
    41 : '建设三路',
    42 : '振宁路',
    43 : '飞虹路',
    44 : '盈丰路',
    45 : '钱江世纪城',
    46 : '钱江路',
    47 : '庆春广场',
    48 : '庆菱路',
    49 : '建国北路',
    50 : '中河北路',
    51 : '凤起路',
    52 : '武林门',
    53 : '沈塘桥',
    54 : '下宁桥',
    55 : '学院路',
    56 : '古翠路',
    57 : '丰潭路',
    58 : '文新',
    59 : '三坝',
    60 : '虾龙圩',
    61 : '三墩',
    62 : '墩祥街',
    63 : '金家渡',
    64 : '白洋',
    65 : '杜甫村',
    66 : '良渚',
    67 : '浦沿',
    68 : '杨家墩',
    69 : '中医药大学',
    70 : '联庄',
    71 : '水澄桥',
    72 : '复兴路',
    73 : '南星桥',
    74 : '甬江路',
    75 : '城星路',
    76 : '市民中心',
    77 : '江锦路',
    78 : '景芳',
    79 : '新塘',
    80 : '新风'
}

with open('../odflow_1s.pk', 'rb') as file:   #用with的优点是可以不用写关闭文件操作
    odflow_1s = pk.load(file)

def pksave(data, file_path):
    file = open(file_path, 'wb')
    pk.dump(data, file)
    file.close()
    return   
    
    
def delta_adj_generate(data_range):
    num_sensors = 81
    od_adj = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    od_adj[:] = 0
    for i in range(data_range.shape[0]):
        time_list=data_range[i]
        if str(time_list) in list(odflow_1s.keys()):
            for j in odflow_1s[str(time_list)].keys():
                startid = stationID(j)
                for k in odflow_1s[str(time_list)][j]:
                    endid = stationID(k)
                    value = odflow_1s[str(time_list)][j][k]
                    od_adj[startid,endid]+=value
    return od_adj

def generate_odadj(freq):
    freq_str=str(freq)+'s'
    starttime = datetime.now() 
    start = '2019-01-01 06:00:00'
    end = '2019-01-25 23:59:00'
    pdates= pd.date_range(start=start, end=end,freq=freq_str)
    
    adj = {}
    for data_start in pdates:
        time1 = datetime.now() 
        data_freq = '1s'
        data_end = data_start + pd.Timedelta(seconds=freq-1)
        data_range = pd.date_range(start=data_start, end=data_end,freq=data_freq)
        od_adj = delta_adj_generate(data_range)
        adj[data_start]=od_adj
        time2 = datetime.now()
        deltatime = (time2 - time1).seconds
        print('one pdates time uses: ', deltatime, ' seconds')
    endtime = datetime.now()
    alltime = (endtime - starttime).seconds
    print('alltime: ',alltime)
    hour = int(alltime/3600)
    minute = int((alltime-hour*3600)/60)
    second = alltime-hour*3600-minute*60
    print( "time used: ", hour, ' hours ',minute,' minutes', second,' second') 
    return adj

def main():
    od_adj_5m = generate_odadj(300)
    pksave(od_adj_5m, './od_adj_5m.pk')
    
if __name__ == '__main__':
    main()
