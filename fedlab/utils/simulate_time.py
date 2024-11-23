_initialized = False
def initialize_once():
    global _initialized
    if not _initialized:
        # 模拟当前的时间
        global cur_time
        cur_time = 0
        _initialized = True
initialize_once()

def get_cur_time():
    return cur_time

def add_cur_time(interval):
    global cur_time
    cur_time += interval
    return cur_time