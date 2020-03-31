import configparser

# Importing Logging config
from os import path
import logging.config
log_file_path = path.join(path.dirname(path.abspath(__file__)), './config/config.cfg')
logging.config.fileConfig(log_file_path)

# from utils import A, B
# from models import TestA
import pandas as pd
import math
import numpy as np
from data.resource_data import CollectData

if __name__ == "__main__":
    df = pd.DataFrame([['1', 2], ['2', 3], ['3', 4], ['4', 5], ['5', 6]], columns=['time', 'value'])

    _save_start = 5
    _data_start = 120
    if _data_start > _save_start:
        # 补上缺失数据
        miss_data_count = math.ceil((_data_start - _save_start) / 60) - 1
        if miss_data_count > 0:
            _time = range(_save_start+60, _data_start, 60)
            df_tmp = pd.DataFrame({'time': _time, 'value': [np.nan for _ in range(miss_data_count)]})
            df = pd.concat([df_tmp, df], axis=0)
            
    else:
        # 去重
        repeat_data_count = math.ceil((_save_start - _data_start) / 60) + 1
        df = df[repeat_data_count:]
        
    df.to_csv("./dataset/test.csv", index=False, header=False, mode="a")

    # _cpu_df.to_csv(_file, index=False, mode='a', header=False)
    # cpudata = CollectData()
    # cpudata.get_config()