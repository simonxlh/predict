import logging
logger = logging.getLogger('usual')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from utils.handle_time import get_start_of_min
from utils.handle_file import get_first_last_line

# 将数据 归一化输出

def get_workload_data(workload_type:str, workload:str, namespace:str, start=None, end=None):
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


if __name__ == "__main__":
    a = get_workload_data("","","",2,60)
    print(a)
    pass

# 将训练数据 转化为模型的输入 放到模型当中