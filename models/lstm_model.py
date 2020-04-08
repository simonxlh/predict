import logging
logger = logging.getLogger('usual')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from matplotlib import pyplot

#Packages for pre processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Importing the Keras libraries and packages for LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load model and rename model learning_rate
from keras.models import load_model
import h5py
import pickle

from data.handle_data import get_all_data
from data.resource_data import ResourceQuery
from config.default_conf import PRED_PERIOD, LSTM_IN_LEN, LSTM_UNITS,\
     LSTM_BATH_SIZE, LSTM_EPOCHS
from models.model_struct import PredictModel

PRED_PERIOD = 1

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=30, n_out=30):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-n_out-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		b = dataset[(i + look_back): (i+look_back+n_out), 0]
		dataY.append(b)
	return np.array(dataX), np.array(dataY)

def data_split(inputs, outputs,val_ratio=0.20):
    '''将输入输出 划分为训练集和验证集
    '''
    val_len = int(len(ins)*val_ratio)

    train_inputs = inputs[:-val_len]
    val_inputs = inputs[-val_len:]
    train_outputs = outputs[:-val_len]
    val_outputs = outputs[-val_len:]
    return train_inputs, val_inputs, train_outputs, val_outputs

    
class LSTMModel(PredictModel):
    _checkpoints_dir = "./checkpoints/"

    def __init__(self):
        pass

    def _preprocess_data(self, data_origin):
        """ 将 data_origin 进行预处理，获得模型的输入输出
        将 data_origin 通过窗口移动，得到输入输出，将问题转化为监督问题

        Args:
            data_origin: shape 为 (times, workload_nums) 的二维数组
        Returns:
            (inputs, outputs)

        """
        data_origin = np.nan_to_num(data_origin)

        max_min = MinMaxScaler()
        data_norm = max_min.fit_transform(data_origin)
        data = data_norm.T

        pkl_path = self._checkpoints_dir + "lstm_scaler.pkl"
        try:
            with open(pkl_path, "wb+") as f:
                pickle.dump(max_min, f)
        except:
            logger.error("MinMaxScaler model save error")

        inputs, outputs = [], []
        # steps = 1
        steps = int(LSTM_IN_LEN/3)

        for i in range(0, data.shape[1] - (LSTM_IN_LEN+PRED_PERIOD+1), steps):
            a = data[:, i:(i+LSTM_IN_LEN)]
            inputs.append(a)

            b = data[:, (i+LSTM_IN_LEN):(i+LSTM_IN_LEN+PRED_PERIOD)]
            outputs.append(b)
        
        return np.array(inputs), np.array(outputs)
    
    def _preprocess_data_one_step(self, data_origin):
        data_origin = np.nan_to_num(data_origin)

        max_min = MinMaxScaler()
        data_norm = max_min.fit_transform(data_origin)

        pkl_path = self._checkpoints_dir + "lstm_scaler.pkl"
        try:
            with open(pkl_path, "wb+") as f:
                pickle.dump(max_min, f)
        except:
            logger.error("MinMaxScaler model save error")

        inputs, outputs = [], []
        # steps = 1
        steps = int(LSTM_IN_LEN/3)

        for i in range(0, len(data_norm) - (LSTM_IN_LEN+PRED_PERIOD), steps):
            a = data_norm[i:(i+LSTM_IN_LEN), :]
            inputs.append(a)

            b = data_norm[(i+LSTM_IN_LEN):(i+LSTM_IN_LEN+PRED_PERIOD), :]
            outputs.append(b)
        
        return np.array(inputs), np.array(outputs)


    def _preprocess_data_reserve(self, data):
        """ 将 data 进行预处理，获得模型的输入输出
        将 data 通过窗口移动，得到输入输出，将问题转化为监督问题

        Args:
            data: shape 为 (workload_nums, times) 的二维数组
        Returns:
            (inputs, outputs)

        """
        inputs, outputs = [], []
        # steps = 1
        steps = int(LSTM_IN_LEN)

        for i in range(0, data.shape[1] - (LSTM_IN_LEN+PRED_PERIOD+1), steps):
            a = data[:, i:(i+LSTM_IN_LEN)]
            inputs.append(a.T)

            b = data[:, (i+LSTM_IN_LEN):(i+LSTM_IN_LEN+PRED_PERIOD)]
            outputs.append(b.T)
        
        return np.array(inputs), np.array(outputs)

    def _save_model(self, resource_query:ResourceQuery, model): 
        # 保存模型
        des_model = "_"+resource_query.name+"_"+\
            str(LSTM_UNITS)+"_"+str(LSTM_BATH_SIZE)+"_"+str(LSTM_EPOCHS)

        _file_name = "./checkpoints/"+"lstm"+des_model+".h5"
        model.save(_file_name)


    def _load_model(self, resource_query:ResourceQuery):
        # 加载模型
        pass

    def _build_model_one_step(self, workloads_num:int):
        '''序列到序列堆叠式LSTM模型 
        '''  
        model=Sequential()  
        model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(LSTM_IN_LEN,workloads_num)))
        model.add(LSTM(256, activation='relu'))
        model.add(Dense(workloads_num))
        model.compile(optimizer='adam', loss='mse')
        return model

    def _build_model(self, input_len):
                # Initialising the RNN
        model = Sequential()
        # Adding the input layerand the LSTM layer
        # model.add(LSTM(units=LSTM_UNITS, activation = 'relu', input_shape=(177, 60))) 
        model.add(LSTM(units=LSTM_UNITS, activation = 'relu', input_shape=(input_len, LSTM_IN_LEN),\
             return_sequences=True))
        # Adding the output layer
        model.add(Dense(units = PRED_PERIOD))
        # Compiling the RNN
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.9, amsgrad=False)

        model.compile(optimizer=adam, loss='mse')
        return model

    def train_model(self, resource_query:ResourceQuery):
        # 获取数据
        df = get_all_data(resource_query)
        if df is None:
            return 

        workloads_list = df.columns
        
        inputs, outputs = self._preprocess_data(df.values)

        # 将数据分为 训练集和验证集
        val_len = int(inputs.shape[0] / 5)
        train_inputs = inputs[:-val_len]
        val_inputs = inputs[-val_len:]
        train_outputs = outputs[:-val_len]
        val_outputs = outputs[-val_len:]

        # print(train_inputs.shape)
        # print(train_outputs.shape)
        model = self._build_model(train_inputs.shape[1])

        # Fitting the RNN to the Training set
        history = model.fit(train_inputs, train_outputs, validation_data=(val_inputs,val_outputs),\
             batch_size=LSTM_BATH_SIZE, epochs=LSTM_EPOCHS, verbose = 2)
        
        self._save_model(resource_query, model)

        self.visual_fit(resource_query, model, val_inputs, val_outputs, workloads_list)

        self.visual_history(resource_query, history)

    def visual_fit(self, resource_query:ResourceQuery, model, val_inputs, val_outputs, workloads_list):

        pass

    
    def visual_history(self, resource_query:ResourceQuery, history):
        # 可视化拟合历史过程

        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])

        pyplot.xlabel('Epoch')
        pyplot.ylabel('Mean Absolute Error Loss')
        pyplot.title('Loss Over Time')
        pyplot.legend(['Train','Valid'])

        des_model = "_"+resource_query.name+"_"+\
            str(LSTM_UNITS)+"_"+str(LSTM_BATH_SIZE)+"_"+str(LSTM_EPOCHS)
        pyplot.savefig("./checkpoints/lstm_history"+des_model+".png")


    def predict(self, start:int, resource_query:ResourceQuery, namespace, workload):
        # 读取模型
        # 利用模型进行预测
        # 预测结果处理后返回
        pass

    def increament_learn(self):
        # 增量训练
        pass

if __name__ == "__main__":
    lstm_model = LSTMModel()
    lstm_model.train_model(ResourceQuery.MEM)