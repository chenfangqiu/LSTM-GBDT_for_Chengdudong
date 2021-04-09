from sklearn.ensemble import GradientBoostingRegressor
import joblib
import pandas as pd
import numpy as np

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

# GBDT数据类型转换
GBDT_X=train_X.reshape(-1,73)
GBDT_Y=train_Y.reshape(-1,1).ravel()
GBDT_test_X=test_X.reshape(-1,73)
GDBT_test_Y=test_Y.reshape(-1,1).ravel()

# 模型构建和训练
gbr = GradientBoostingRegressor(n_estimators=30,learning_rate=0.01)
gbr.fit(GBDT_X,GBDT_Y)
joblib.dump(gbr, 'gbr.pkl')