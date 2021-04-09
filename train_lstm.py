import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


#### 导入数据 ####
#data=pd.read_csv('/data/home/u20120778/test/cddata.csv',usecols=[1])    #学校超算平台目录
data=pd.read_csv('cddata.csv',usecols=[1])
data=data.dropna()
dataset = data.values
dataset = dataset.astype('float32')


#### 归一化处理 ####
max_value = np.max(dataset) 
min_value = np.min(dataset)  
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))


#### 制作lstm训练集 ####
def create_dataset(dataset, look_back=73): # lookback=73，一天有73个十五分钟客流数据，以此类推，要做一周可以选择73*7
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back + 1])
    return np.array(dataX), np.array(dataY)
data_X, data_Y = create_dataset(dataset)
# 除去最后一周的数据
train_size = int(len(data_X)-73*7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
# 数据类型变换
train_X = train_X.reshape(-1, 1, 73) #训练数据转化为1列，一个数据73维
train_Y = train_Y.reshape(-1, 1, 1) #测试数据转化为1列，一个数据1维
train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)


#### 使用torch构建LSTM模型 ####
class lstm(nn.Module):
    def __init__(self,input_size=73,hidden_size=4,output_size=1,num_layer=2): #详见nn.LSTM参数设置页面
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer) #第一层为num_layer层LSTM
        self.layer2 = nn.Linear(hidden_size,output_size) #第二层为全连接层
# 定义前向传输路径    
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x
# 构建模型
model = lstm(73, 4,1,2)
# 判断平台是否支持cuda加速
use_gpu = torch.cuda.is_available()
print(use_gpu)
if use_gpu:
    model = model.cuda()
# 设置标砖和优化方式
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


#### 训练开始 ####
for e in range(1000):
    if use_gpu:
        var_x = Variable(train_x).cuda()
        var_y = Variable(train_y).cuda()
    else:
        var_x = Variable(train_x)
        var_y = Variable(train_y)
# 前向传播
    out = model(var_x)
    loss = criterion(out, var_y)
# 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (e + 1) % 100 == 0: # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

#### 保存训练好的模型 ####
torch.save(model, 'lstm_single.pt')