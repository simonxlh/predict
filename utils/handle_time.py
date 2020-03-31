import logging
logger = logging.getLogger('usual')
import time


def get_start_of_min(stamp=None)-> int:
    """将时间戳精确到分钟

    将传入的时间戳转到对应分钟开始处，如果没有传，则返回当前时间戳精确到分钟

    Args:
        stamp: 需要转换的时间戳，精确到秒级，可以是 float 类型

    Returns:
        int, 精确到分钟的时间戳
    """
    _time = time.time()
    if (not stamp is None) and isinstance(stamp, int):
        _time = stamp
    
    return int(_time/60)*60
