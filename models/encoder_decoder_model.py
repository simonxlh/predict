import logging
logger = logging.getLogger('usual')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.handle_data import get_workload_data
from data.resource_data import ResourceQuery
from config.default_conf import PRED_PERIOD
from models.model_struct import PredictModel

class EncodeDecoderModel(PredictModel):

    def __init__(self):
        pass

    def _preprocess_data(self, data):
        # 数据的预处理
        pass

    def _save_model(self, resource_query:ResourceQuery): 
        # 保存模型
        pass

    def _load_model(self, resource_query:ResourceQuery):
        # 加载模型
        pass

    def train_model(self, resource_query:ResourceQuery):
        # 获取数据
        # 数据预处理
        # 训练模型
        # 保存模型
        pass
    
    def visual_fit(self, resource_query:ResourceQuery):
        # 可视化拟合程度
        pass

    def predict(self, start:int, resource_query:ResourceQuery, namespace, workload):
        # 读取模型
        # 利用模型进行预测
        # 预测结果处理后返回
        pass

    def increament_learn(self):
        # 增量训练
        pass
