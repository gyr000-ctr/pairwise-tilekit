"""tilekit 核心 API（初始未实现）。"""


class Grid:
    def __init__(self, w, h):
        """构造"""
        raise NotImplementedError

    def set_wall(self, x, y):
        """设墙"""
        raise NotImplementedError

    def astar(self, start, goal):
        """寻路"""
        raise NotImplementedError
