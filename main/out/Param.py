DATANAME = 'HZ'
TASK = "out"

#数据采用统计周期
TIME_INTERVAL = 10

#数据开始时间，1日到25日，共25天数据
DATA_START_DAY = 1
DATA_END_DAY = 25

#每天被截取的时间，早晨06:00到晚上23:30
DAYSTART = "06:00"
DAYEND = "23:30"

N_NODE=81
#输入时间步，输出时间步
INPUT_STEP = 12
PRED_STEP = 6
HORIZON = 0

# train test 划分
TRAINRATIO = 0.8 # TRAIN + VAL
TRAINVALSPLIT = 0.25 # val_ratio = 0.8 * 0.125 = 0.1

CHANNEL = 1
BATCHSIZE = 8
LEARN = 0.001
EPOCH = 200
PATIENCE = 20
OPTIMIZER = 'Adam'
LOSS = 'MAE'
# LOSS = 'MSE'
# LOSS = 'MAEMAPE'

CL_DELAY_STEPS = 2000

# used in debug
ADJPATH = './data/adj/W_od.csv'
ADJTYPE = 'doubletransition'