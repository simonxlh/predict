# 仅作为新增 model 的模板

from abc import abstractmethod, ABCMeta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.resource_data import ResourceQuery

class PredictModel(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def _preprocess_data(self, data):
        # 数据的预处理
        pass

    @abstractmethod
    def _save_model(self, resource_query:ResourceQuery, model): 
        # 保存模型
        pass

    @abstractmethod
    def _load_model(self, resource_query:ResourceQuery):
        # 加载模型
        pass

    @abstractmethod
    def train_model(self, resource_query:ResourceQuery):
        # 获取数据
        # 数据预处理
        # 训练模型
        # 保存模型
        pass
    
    @abstractmethod
    def visual_fit(self, resource_query:ResourceQuery):
        # 可视化拟合程度
        pass

    @abstractmethod
    def predict(self, start:int, resource_query:ResourceQuery, namespace, workload):
        # 读取模型
        # 利用模型进行预测
        # 预测结果处理后返回
        pass

    @abstractmethod
    def increament_learn(self):
        # 增量训练
        pass

