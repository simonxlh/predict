''' 只用作对比实验，实际应用过程中需要针对不同负载训练不同模型
    获取训练数据、验证数据
    训练模型
    保存模型
    读取模型
    图形化损失（分为训练数据、验证数据）
    接收数据、返回预测结果
 '''
import logging
logger = logging.getLogger('usual')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import time
import gc
import pandas as pd
from matplotlib import pyplot
from fbprophet import Prophet

from data.handle_data import get_workload_data
from data.resource_data import ResourceQuery
from config.default_conf import PRED_PERIOD
from models.model_struct import PredictModel


class ProphetModel(PredictModel):
    '''使用 FaceBook 的 Prophet 进行预测
       由于需要训练多个模型才能预测多种负载，该方案只作为对比实验
       对比实验中，使用 dongwu 下的 Deployment:compass-backend-latest 这一负载
    '''
    _namespace = "dongwu"
    _workload = "Deployment:compass-backend-latest"
    holidays_prior_scale = 20
    
    def __init__(self):
        pass

    def _preprocess_data(self):
        df = get_workload_data(ResourceQuery.CPU, "Deployment:compass-backend-latest", "dongwu")
        df.rename(columns={'Deployment:compass-backend-latestdongwu':'y'}, inplace = True)
        df["ds"] = df.index
        
        # dataframe 时间戳 => `%Y-%m-%d %H:%M` 时间
        df['ds'] = df['ds'].apply(lambda x:time.strftime("%Y-%m-%d %H:%M", time.localtime(x)))

        return df

    def _save_model(self, resource_query:ResourceQuery, model):
        # pkl_path 模型保存地址
        pkl_path = "./checkpoints/Prophet.pkl"

        try:
            with open(pkl_path, "wb") as f:
            # Pickle the 'Prophet' model using the highest protocol available.
                pickle.dump(model, f)
        except:
            logger.error("prophet model save error")
            

    def _load_model(self, resource_query:ResourceQuery):
        pkl_path = "./checkpoints/Prophet.pkl"

        try:
            with open(pkl_path, 'rb') as f:
                m = pickle.load(f)
        except:
            logger.error("prophet model load error")
            return None

        return m

    def get_holidays(self):
        weekends = pd.DataFrame({
            'holiday': 'weekends',
            'ds': pd.to_datetime(['2020-03-01', '2020-03-08', '2020-03-15',
                                    '2020-03-22', '2020-04-05', '2020-04-12',]),
            'lower_window': -1,
            'upper_window': 1,
        })

        # holidays = pd.concat((playoffs, superbowls))
        return weekends

    def train_model(self, resource_query:ResourceQuery):
        # 获取处理后的数据
        df = self._preprocess_data()
        print(df.head())

        # 添加假期
        holidays = self.get_holidays()

        # 训练模型
        m = Prophet(interval_width=0.85,weekly_seasonality=True,\
             holidays=holidays, holidays_prior_scale=self.holidays_prior_scale)
        # m = Prophet(interval_width=0.95,weekly_seasonality=True)
        m.fit(df)
        # future = m.make_future_dataframe(periods=240, freq='min')
        # m.predict(future)

        # 保存模型
        self._save_model(resource_query, m)


    def visual_fit(self, resource_query:ResourceQuery):

        # 获取原始数据
        df = self._preprocess_data()
        
        # 读取模型
        model = self._load_model(resource_query)
        if model is None:
            return None
        
        future = model.make_future_dataframe(periods=60, freq='min')
        
        # 获得拟合数据
        forecast = model.predict(future)

        # 绘制图像
        fit_x = forecast['ds']
        origin_x = fit_x.iloc[:-60]

        fit_y = forecast['yhat']
        origin_y = df['y']

        pyplot.plot(origin_x, origin_y, color="blue", label="origin_data")
        pyplot.plot(fit_x, fit_y, color="red", label="fit_data")
        pyplot.legend(loc='upper left')
        pyplot.savefig("./checkpoints/prophet_"+str(self.holidays_prior_scale)+".png")


    def predict(self, start, resource_query:ResourceQuery, namespace=_namespace, workload=_workload):
        # 生成需要预测的时间区间
        ds_df = pd.DataFrame(data=list(time.strftime("%Y-%m-%d %H:%M", time.localtime(x)) \
            for x in range(start, start+60*(PRED_PERIOD+1), 60)), columns=["ds"])
        
        # 读取模型
        model = self._load_model(resource_query)
        if model is None:
            return None
        
        # 利用模型进行预测
        forecast = model.predict(ds_df)

        _tmp = forecast.loc[:,["ds", "yhat"]]

        # 预测结果处理后返回
        _tmp["ds"] = _tmp["ds"].apply(lambda x:\
            int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))

        del forecast, model, ds_df
        gc.collect()
        
        return _tmp.values
    
    def increament_learn(self):
        # 增量训练
        pass

if __name__ == "__main__":
    prophet_model = ProphetModel()
    prophet_model.train_model(ResourceQuery.MEM)
    prophet_model.visual_fit(ResourceQuery.MEM)