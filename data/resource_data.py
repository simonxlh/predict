"""Handle CPU data from prometheus monitoring"""

import requests
import math
import gc
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger('usual')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# time scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

from enum import Enum 

from utils.handle_time import get_start_of_min
from utils.handle_file import get_first_last_line
from config.default_conf import MO_ACCESS_URI,MO_RETENTION,MO_COLLECT_STEP

class ResourceQuery(Enum):
    CPU = "namespace:workload_cpu_usage:sum"
    MEM = "namespace:workload_memory_usage:sum"


class CollectData(object):
    """主要负责收集监控的数据
    """

    # 由于 sys.path 添加了上级目录的path，这里保存文件从添加的path开始寻找
    _save_file_dir = "./dataset/"

    def __init__(self):
        
        if self.check_status():
            mem_start, mem_end = self._get_time_period(ResourceQuery.MEM)
            cpu_start, cpu_end = self._get_time_period(ResourceQuery.CPU)

            self._query_range_info(resource_query=ResourceQuery.MEM,\
                start=mem_start, end=mem_end, step=60)
            self._query_range_info(resource_query=ResourceQuery.CPU,\
                start=cpu_start, end=cpu_end, step=60)
            sched = BlockingScheduler()
            sched.add_job(self._interval_job, 'interval', minutes=MO_COLLECT_STEP)
            sched.start()

    def check_status(self):
        """获取当前监控状态

        Returns: whether monitoring is available
        """
        _request_url = MO_ACCESS_URI + '/api/v1/targets'
        response = requests.request('GET', _request_url)
        if response.status_code == 200:
            logger.info("monitoring is active")
            return True
        else:
            logger.error("monitoring is inactive")
        return False

    def _get_time_period(self, resource_query:ResourceQuery) -> (int or str, int):
        """获取请求的时间区间
        
        根据是否有保存的数据来确定请求的时间区间。
        有，则开始时间 start 使用最后一条数据所处的时间戳；
        无，则使用监控保存的时长。
        结束时间 end 统一使用当前时间戳（精确到分钟）

        Args:
        resource_query: ResourceQuery 的枚举类型

        Returns:
            tuple(start, end)
        """
        file_name = resource_query.name
        time_file = self._save_file_dir + file_name + ".csv"
        end = get_start_of_min()
        start = end - MO_RETENTION*24*60*60
        if not os.path.exists(time_file):
            # 尚未有保存的数据
            return (start, end)

        # 获取记录时间点文件最后一行
        _, _line = get_first_last_line(time_file)
        if not _line is None:
            start = _line.strip().split(',')[0]
        return (start, end)

    def _query_range_info(self, resource_query:ResourceQuery, **kwargs):
        """根据 query 表达式向 prometheus 来获取查询结果，并进行处理
        
        由于 prometheus 返回结果有 points 数量的限制，如果结果数量超出范围将启用二分查询。

        Args:
            resource_query: ResourceQuery 的枚举类型
            kwargs: 包括 查询开始时间 start, 结束时间 end, 查询步长 step
        """
        kwargs['query'] = resource_query.value
        r = requests.get(MO_ACCESS_URI+'/api/v1/query_range', params=kwargs)
        if r.status_code == 400:
            data = r.json()

            if data['error'].find('exceeded maximum resolution') >= 0:
                mid = int((kwargs['start'] + kwargs['end'])/2)
                logger.info("too many points, start binary query!")
                self._query_range_info(resource_query=resource_query, start=kwargs['start'], end=mid,\
                     step=kwargs['step'])
                self._query_range_info(resource_query=resource_query, start=mid, end=kwargs['end'],\
                     step=kwargs['step'])
            else:
                logger.error("monitoring has error about"+ data['error'])
            
        elif r.status_code == 200:
            logger.info("handle " + resource_query.value +" from " + str(kwargs['start'])\
                + " to "+ str(kwargs['end']))
            print('start: %s, end: %s' % (kwargs['start'],kwargs['end']))

            self._handle_workload_data(r.json()['data']['result'], resource_query,\
                 int(kwargs['start']), int(kwargs['end']))

            # if resource_query is ResourceQuery.MEM:
            #     self._handle_workload_data(r.json()['data']['result'], ResourceQuery.MEM)
            # elif resource_query is ResourceQuery.CPU:
            #     self._handle_workload_data(r.json()['data']['result'], ResourceQuery.CPU)
            # else:
            #     pass

        del r
        gc.collect()

    def _interval_job(self):
        """定时向 Prometheus 获取新的监控数据
        """
        mem_start, mem_end = self._get_time_period(ResourceQuery.MEM)
        cpu_start, cpu_end = self._get_time_period(ResourceQuery.CPU)
        self._query_range_info(resource_query=ResourceQuery.MEM,\
            start=mem_start, end=mem_end, step=60)
        self._query_range_info(resource_query=ResourceQuery.CPU,\
            start=cpu_start, end=cpu_end, step=60)

    def _handle_workload_data(self, data, resource_query:ResourceQuery,\
         start:int, end:int):
        workload_type = resource_query.name
        print(workload_type + ' worload data len :',len(data))

        _file_name = self._save_file_dir+workload_type+".csv"
        # 初始化为空的 DataFrame
        origin_df = pd.DataFrame()

        if os.path.exists(_file_name):
            # 原本有保存的 csv 则进行读取
            origin_df = pd.read_csv(_file_name, index_col=0)
        
        _tmp_df_all = pd.DataFrame(index=list(x for x in range(start, end+60, 60)))
        for index in range(0,len(data)):
            _data = data[index]
            _metric = _data['metric']
            _spec_name = _metric['workload'] + _metric['namespace']
            _data_value = np.array(_data['values'])

            # 注意请求中返回的时间戳为 "1586216580" 型的字符串，需要通过 list(map(int,index)) 转为 int
            _tmp_df = pd.DataFrame(data=_data_value[:,1], columns=[_spec_name], index=list(map(int, _data_value[:,0])))
            
            _tmp_df_all = pd.concat([_tmp_df_all, _tmp_df], axis=1)

            # Garbage Collection
            del _data, _metric, _spec_name, _data_value, _tmp_df
            gc.collect()
        
        # 将 origin_df 与 _tmp_df_all 进行拼接
        origin_df = pd.concat([origin_df, _tmp_df_all], sort=True)

        # 根据重复索引 保留最后一次出现的值
        origin_df = origin_df[~origin_df.index.duplicated(keep="last")]

        origin_df.to_csv(_file_name, mode='w', header=True, index=True)

        del origin_df, _tmp_df_all, _file_name
        gc.collect()

# 分文件存储的逻辑
    def _get_time_period_split(self) -> (int or str, int):
        """获取请求的时间区间
        
        适用于存储的监控数据为以负载名为文件名的格式

        Returns:
            tuple(start, end)
        """
        time_file = self._save_file_dir + "time.csv"
        end = get_start_of_min()
        start = end - MO_RETENTION*24*60*60
        if not os.path.exists(time_file):
            # 尚未有保存的数据
            return (start, end)

        # 获取记录时间点文件最后一行
        _, _line = get_first_last_line(time_file)
        if not _line is None:
            start = _line.strip().split(',')[1]
        return (start, end)

    def _handle_workload_data_split(self, data, workload_type:str):
        
        print(workload_type + ' worload data len :',len(data))
        for index in range(0,len(data)):
            _data = data[index]
            _metric = _data['metric']
            _spec_name = _metric['workload'] + _metric['namespace'] +"_"+workload_type+".csv"
            _cpu_df = pd.DataFrame(data=_data['values'], columns=('time', 'value'))
            # # 是否有历史数据
            _file = self._save_file_dir + _spec_name
            _saved = os.path.exists(_file)

            if not _saved:
                _cpu_df.to_csv(_file, index=False, mode='w', header=False)
            else:
                # 处理定时任务带来的 重复的数据 或 缺失的数据
                # 获取到存储的最后一行的时间 与当前 dataframe 的开始处时间进行比较
                _, _line = get_first_last_line(_file)
                _save_start = int(_line.strip().split(',')[0])
                _data_start = int(_data['values'][0][0])
                if _data_start > _save_start:
                    # 补上缺失数据
                    miss_data_count = math.ceil((_data_start - _save_start) / 60) - 1
                    if miss_data_count > 0:
                        _time = range(_save_start+60, _data_start, 60)
                        df_tmp = pd.DataFrame({'time': _time, 'value': [np.nan for _ in range(miss_data_count)]})
                        _cpu_df = pd.concat([df_tmp, _cpu_df], axis=0, sort=True)

                        # Garbage Collection
                        del df_tmp
                        gc.collect()
                else:
                    # 去重
                    repeat_data_count = math.ceil((_save_start - _data_start) / 60) + 1
                    _cpu_df = _cpu_df[repeat_data_count:]
                
                _cpu_df.to_csv(_file, index=False, mode='a', header=False)
            
            # Garbage Collection
            del _cpu_df
            gc.collect()

    def get_config(self):
        print(MO_ACCESS_URI)
        print(MO_RETENTION)
        print(MO_COLLECT_STEP)


if __name__ == "__main__":
    cpudata = CollectData()
