# 装饰器- 计时+捕获异常
import time
import traceback
from functools import wraps


def time_wrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        print("{0}-任务开始".format(func_name))
        start = time.time()
        res = None
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
        use_time = time.time() - start
        print("{0}-任务执行完毕，用时:{1:.2f}，".format(func_name, use_time))
        return res
    return wrapper
