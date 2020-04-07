import logging
logger = logging.getLogger('usual')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from utils.handle_time import get_start_of_min
from utils.handle_file import get_first_last_line
from data.resource_data import ResourceQuery

def get_all_data(resource_query:ResourceQuery):
    """获取对应资源类型的所有信息
    
    Args:
        resource_query: ResourceQuery 枚举类型

    """
    _file_name = "./dataset/"+resource_query.name+".csv"
    if os.path.exists(_file_name):
        # 原本有保存的 csv 则进行读取
        origin_df = pd.read_csv(_file_name, index_col=0)
        return origin_df
    else:
        logger.error(resource_query.name+" not exist")
        return None



def get_workload_data(resource_query:ResourceQuery, workload:str, namespace:str, start=None, end=None):
    """获取对应负载资源使用的历史信息
    
    使用 type workload namespace 相结合，确定一个唯一的负载资源使用
    取出 [start, end] 区间的数据， start 未指定则默认为开始处，end 未指定则默认为结束处


    Args:
        resource_query: ResourceQuery 枚举类型
        workload: prometheus 中的 workload,
            格式为 ${workload_type}:${workload_name}，例如 Deployment:console-ping-latest
        namespace: 负载资源所处的命名空间
    Returns:
        [start, end] 区间的 DataFrame or None
    """
    _file_name = "./dataset/"+resource_query.name+".csv"
    _spec_name = workload + namespace
    origin_df = pd.DataFrame()

    # start , end 未必是分钟开始处 获取开始时间
    first_line , last_line = get_first_last_line(_file_name)

    # 获取负载所在第几列，记为 _index，捕获异常则没有该列
    try:
        _index = first_line.strip().split(',').index(_spec_name)
    except:
        _index = -1
        return None

    if os.path.exists(_file_name):
            # 原本有保存的 csv 则进行读取
            origin_df = pd.read_csv(_file_name, index_col=0, usecols=[0, _index])
    else:
        logger.error("resoruce: "+ workload +" "+resource_query.name+" not exist")
        return None

    # 当开始、结束时间为 None 时，确定负载真实的开始时间（第一个非 NaN 所在行对应索引）、结束时间
    if not start is None:
        start = get_start_of_min(start)
    else:
        _arr = origin_df[origin_df[_spec_name].notna()].index
        if len(_arr) == 0:
            return None
        else:
            start = _arr[0]
    
    if not end is None:
        end = get_start_of_min(end)
    else:
        end = int(last_line.strip().split(',')[0])
    
    return origin_df[(origin_df.index <= end) & (origin_df.index >= start)]


def get_workload_data_split(workload_type:str, workload:str, namespace:str, start=None, end=None):
    """获取对应负载资源使用的历史信息
    
    使用 type workload namespace 相结合，确定一个唯一的负载资源使用
    取出 [start, end] 区间的数据， start 未指定则默认为开始处，end 未指定则默认为结束处


    Args:
        workload_type: "cpu" or "mem"
        workload: prometheus 中的 workload,
            格式为 ${workload_type}:${workload_name}，例如 Deployment:console-ping-latest
        namespace: 负载资源所处的命名空间
    Returns:
        [start, end] 区间的 DataFrame or None
    """
    _file = "./dataset/"+workload+namespace+"_"+workload_type+".csv"
    _file = "./dataset/test.csv"
    _df = pd.read_csv(_file, header=None, names=['time', 'value'], usecols=[0,1])
    # start , end 未必是分钟开始处
    first_line , last_line = get_first_last_line(_file)
    if not start is None:
        start = get_start_of_min(start)
    else:
        start = int(first_line.strip().split(',')[0])
    if not end is None:
        end = get_start_of_min(end)
    else:
        end = int(last_line.strip().split(',')[0])

    return _df[(_df['time']>=start) &(_df['time']<=end)]

# 将数据 归一化输出

if __name__ == "__main__":
    a = get_workload_data("","","",2,60)
    print(a)
    pass

