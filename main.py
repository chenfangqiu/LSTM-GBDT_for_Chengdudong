import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt
from sklearn import metrics

#### 使用torch构建LSTM模型 ####
class lstm(nn.Module):
    def __init__(self,input_size=73,hidden_size=4,output_size=1,num_layer=2): #详见nn.LSTM参数设置页面
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer) #第一层为num_layer层LSTM
        self.layer2 = nn.Linear(hidden_size,output_size) #第二层为线性
# 定义前向传输路径    
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x

#data=pd.read_csv('/data/home/u20120778/test/data.csv',usecols=[5])
data=pd.read_csv('cddata.csv',usecols=[1])
data=data.dropna()
dataset = data.values   # 获得csv的值
dataset = dataset.astype('float32')
max_value = np.max(dataset)  # 获得最大值
min_value = np.min(dataset)  # 获得最小值
scalar = max_value - min_value  # 获得间隔数量
dataset = list(map(lambda x: x / scalar, dataset)) # 归一化

def create_dataset(dataset, look_back=73):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# 创建好输入输出
data_X, data_Y = create_dataset(dataset)

# 划分训练集和测试集
train_size = int(len(data_X)-73)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

# LSTM数据类型变换
train_X = train_X.reshape(-1, 1, 73)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 73)
train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)

# GBDT数据类型转换
GBDT_X=train_X.reshape(-1,73)
GBDT_Y=train_Y.reshape(-1,1).ravel()
GBDT_test_X=test_X.reshape(-1,73)
GDBT_test_Y=test_Y.reshape(-1,1).ravel()

# LSTM预测
var_data = Variable(test_x)
new_m = torch.load('lstm_200.pt')
predict = new_m(var_data)
predict = predict.view(-1).data.numpy()

# GBDT预测
gbr = joblib.load('gbr.pkl')
y_gbr = gbr.predict(GBDT_test_X)
test_score = np.zeros((3000,), dtype=np.float64)
y_pre=[]
for i, y_pred in enumerate(gbr.staged_predict(GBDT_test_X)):
    test_score[i] = gbr.loss_(GDBT_test_Y, y_pred)
    y_pre.append(y_pred)

# LSTM效果评价
x1=torch.from_numpy(np.array(predict).reshape(-1,1,73))
x2=torch.from_numpy(np.array(dataset[:73*1]).reshape(-1,1,73))
criterion = nn.MSELoss()
loss = criterion(x1,x2)
print('LSTM-MSE：'+str(loss.item())) # MSE
print('LSTM-R2_SCORE：'+str(metrics.r2_score(dataset[:73*1], predict))) # r2_score

# GBDT效果评价
x1=torch.from_numpy(np.array(y_pre[2999]).reshape(-1,1,73))
x2=torch.from_numpy(np.array(dataset[:73*1]).reshape(-1,1,73))
criterion = nn.MSELoss()
loss = criterion(x1,x2)
print('GBDT-MSE：'+str(loss.item())) # MSE
print('GBDT-R2_SCORE：'+str(metrics.r2_score(dataset[:73*1], y_pre[2999]))) # r2_score

# 作图
plt.plot(dataset[:73*1], 'b', label='REAL')
plt.plot(predict, 'r', label='LSTM')
plt.plot(y_pre[2999], 'g', label='GBDT')
plt.legend(loc='best')
plt.show()
#plt.savefig('result.png')