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

from data.handle_data import get_all_data
from data.resource_data import ResourceQuery
from config.default_conf import PRED_PERIOD, LSTM_IN_LEN, LSTM_UNITS,\
     LSTM_BATH_SIZE, LSTM_EPOCHS
from models.model_struct import PredictModel

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=30, n_out=30):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-n_out-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		b = dataset[(i + look_back): (i+look_back+n_out), 0]
		dataY.append(b)
	return np.array(dataX), np.array(dataY)

class LSTMModel(PredictModel):

    def __init__(self):
        pass


    def _preprocess_data(self, data):
        """ 将 data 进行预处理，获得模型的输入输出
        将 data 通过窗口移动，得到输入输出，将问题转化为监督问题

        Args:
            data: shape 为 (workload_nums, times) 的二维数组
        Returns:
            (inputs, outputs)

        """
        inputs, outputs = [], []
        # steps = 1
        steps = int(LSTM_IN_LEN/3)

        for i in range(0, data.shape[1] - (LSTM_IN_LEN+PRED_PERIOD+1), steps):
            a = data[:, i:(i+LSTM_IN_LEN)]
            inputs.append(a)

            b = data[:, (i+LSTM_IN_LEN):(i+LSTM_IN_LEN+PRED_PERIOD)]
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

    def train_model(self, resource_query:ResourceQuery):
        # 获取数据
        df = get_all_data(resource_query)
        if df is None:
            return 
        inputs, outputs = self._preprocess_data_reserve(df.values.T)

        # 将数据分为 训练集和验证集
        val_len = int(inputs.shape[0] / 5)
        train_inputs = inputs[:-val_len]
        val_inputs = inputs[-val_len:]
        train_outputs = outputs[:-val_len]
        val_outputs = outputs[-val_len:]

        train_inputs = np.array(inputs[0]).reshape((1,60,170))
        train_outputs = np.array(outputs[0]).reshape((1,60,170))
        val_inputs = np.array(inputs[1]).reshape((1,60,170))
        val_outputs = np.array(outputs[1]).reshape((1,60,170))

        print(train_inputs.shape)
        print(train_outputs.shape)

        # Initialising the RNN
        model = Sequential()
        # Adding the input layerand the LSTM layer
        model.add(LSTM(units=LSTM_UNITS, activation = 'relu', input_shape=(LSTM_IN_LEN, 170),\
             return_sequences=True))
        # Adding the output layer
        model.add(Dense(units = 170))
        # Compiling the RNN
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        # Fitting the RNN to the Training set
        history = model.fit(train_inputs, train_outputs, validation_data=(val_inputs,val_outputs),\
             batch_size=LSTM_BATH_SIZE, epochs=LSTM_EPOCHS, verbose = 2)
        
        self._save_model(resource_query, model)

        self.visual_fit(resource_query, history)

    
    def visual_fit(self, resource_query:ResourceQuery, history):
        # 可视化拟合程度

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
    lstm_model.train_model(ResourceQuery.CPU)