# compute_hetero: 指只有计算差异，其他理想化
# communicate_hetero: 指只有通信差异，其他理想化
# practical: 指system，statistical都不一样
# statistical_iid: 指只有数据

_initialized = False
def initialize_System_Conf_once():
    global _initialized
    if not _initialized:
        # 模拟当前的时间
        global system_hetero
        system_hetero = None
        _initialized = True
initialize_System_Conf_once()

def get_cur_system_hetero():
    global system_hetero
    return system_hetero

def set_cur_system_hetero(val):
    """
    只能是
    Args:
        val:
            compute_hetero: 指只有计算差异，其他理想化
            communicate_hetero: 指只有通信差异，其他理想化
            practical: 指system，statistical都不一样
    Returns:

    """
    global system_hetero
    system_hetero = val