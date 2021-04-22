# Use LSTM and GBDT to predict the the number of people entering Chengdudong Railway Station

#### Useage
1. Use the model I've trained: python main.py
2. Train the model by yourself:
    1. python train_lstm.py or python train_gbdt.py
    2. get the model file lstm_200.pt and gbr.pkl
    3. python main.py

#### Introduction

Forecasting theory and methods course assignments

#### Train LSTM

1. Import the data, remove the null value with dropna() method, and change the type to float32;
2. Data dimensionless and normalized;
3. Making LSTM training set:
    1. Determine the window size. Since the target is to forecast one-day data, there are 73 15-minute passenger flow data in a day, so the window size is set to 73.
    2. Make data sets of X and Y. Perform 4379 cycles (the number of all data minus the size of the window and then subtract 1). In the ith cycle: place the I to I +73 data of the data set in the I row of the X matrix, and place the I +74 data of the data set in the I bit of the Y vector;
    3. Make X and Y of the training set. Remove the last week's data from X and Y;
    4. Convert the training data into Tensor, X into Tensor of 4307 row and 1 column with 73 dimensions, Y into Tensor of 4307 row and 1 column with 1 dimension;
4. Construction of LSTM model:
    1. The input dimension (input_size) is 73. Other parameters can be adjusted by themselves. In this experiment, hidden_size=4,output_size=1,num_layer=2, and the other parameters remain the default values in the torch.
    2. The first layer of the network structure is set as a two-layer LSTM, and the second layer is the full connection layer;
    3. Set the loss function as MSE and the optimizer as ADAM, and the learning rate is 0.1;
5. Model training, you can choose the number of iterations, 1000 times in this experiment;
6. Forecast and evaluate.

#### Train GBDT

1. Same as LSTM's 1 through 3 (Tensor replaced with Array);
2. Build GBDT regression tree model, the number of weak learners is 3000 (other number can be selected), the learning rate is 0.01, and the other parameters are consistent with the default value of Scikit-Learn;
3. Model training;
4. Forecast and evaluate.

# 分别使用LSTM和GBDT预测成都东站进站客流

#### 使用说明

1. 使用本实验已经训练好了的模型：直接运行main.py
2. 自行训练：  
2.1. 运行train_lstm.py和train_gbdt.py，得到lstm_200.pt和gbr.pkl两个模型文件.   
2.2. 运行main.py. 

#### 介绍

预测理论与方法2021春季课程作业

#### LSTM部分
1、导入数据，利用dropna()方法去除空值，将类型转变为float32;  
2、数据去量纲，归一化处理;  
3、制作LSTM训练集：  
    3.1、确定窗口大小。由于目标是预测一日数据，一日有73个十五分钟客流数据，故窗口大小设定为73；  
    3.2、制作X和Y的数据集。进行4379次（所有数据数量减窗口大小再减1）循环，在第i次循环中：将数据集的第i到i+73个数据放于X矩阵的第i行中，并将数据集的第i+74个数据放于Y向量的第i位中；  
    3.3、制作训练集的X和Y。去除X和Y中的最后一周数据；  
    3.4、将训练数据转化为tensor，X转变为4307行1列73维的tensor，Y转变为4307行1列1维的tensor；  
4、构建LSTM模型：  
    4.1、输入维度（input_size）为73维、其他参数可以自行调整，本实验中使用hidden_size=4,output_size=1,num_layer=2，其余参数保持torch中的默认值；  
    4.2、网络结构第一层设定为双层LSTM，第二层为全连接层；  
    4.3、设定损失函数为MSE、优化器为Adam，学习率0.1；  
5、模型训练，可以自选迭代次数，本实验中选择1000次；  
6、预测并评价。  

#### GBDT部分
1、与LSTM的1到3相同，但不转化为tensor，转化为array；  
2、构建GBDT回归树模型，弱学习器数量为3000个（也可选择别的数量），学习率0.01，其余参数与scikit-learn默认值一致；  
3、模型训练；  
4、预测并评价。  
