"""tilekit 核心 API：二维网格与 A* 寻路。

坐标采用 ``(x, y)``，x 为列、y 为行，原点在左上角。移动仅限上下左右
四邻域，每一步代价为 1，因此 A* 返回的路径步数等于 BFS 最短路。

实现要点：
* 墙体保存在带一圈边界的 ``bytearray`` 中，边界恒为墙，邻居扩展无需做
  越界判断；内存为 O(格数)。
* 每次 ``astar`` 的状态（g 值、父节点、开放表）都是调用内的局部变量，
  多个线程并发地对同一 ``Grid`` 寻路互不干扰；墙体在寻路开始时于锁内
  做一次快照，并与 ``set_wall`` / ``clear_wall`` 同步。
* 开放表是调用内按 ``f = g + Manhattan`` 分桶的 Dial 队列：f 单调不减
  且为整数，入队出队都是 O(1)，只占用 O(格数) 内存。不含任何随机量，
  结果确定可复现。
"""

from __future__ import annotations

import threading
from array import array

__all__ = ["Grid"]

_WALL = 1
_CLOSED = -1


class Grid:
    """二维网格，``w`` 列 ``h`` 行。"""

    __slots__ = ("_w", "_h", "_stride", "_walls", "_lock")

    def __init__(self, w, h):
        if type(w) is not int or type(h) is not int:
            raise ValueError("w and h must be integers")
        if w <= 0 or h <= 0:
            raise ValueError("w and h must be positive")
        stride = w + 2
        buf = bytearray((h + 2) * stride)
        # 上下两行与左右两列设为墙，充当越界哨兵。
        buf[0:stride] = b"\x01" * stride
        buf[(h + 1) * stride:(h + 2) * stride] = b"\x01" * stride
        buf[0::stride] = b"\x01" * (h + 2)
        buf[stride - 1::stride] = b"\x01" * (h + 2)
        self._w = w
        self._h = h
        self._stride = stride
        self._walls = buf
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 墙体维护
    # ------------------------------------------------------------------
    def _index(self, x, y, what):
        if type(x) is not int or type(y) is not int:
            raise ValueError("%s coordinates must be integers" % what)
        if not (0 <= x < self._w and 0 <= y < self._h):
            raise ValueError("%s coordinate out of bounds" % what)
        return (y + 1) * self._stride + (x + 1)

    def set_wall(self, x, y):
        """把 ``(x, y)`` 设为墙；坐标越界时抛 ``ValueError``。"""
        idx = self._index(x, y, "wall")
        with self._lock:
            self._walls[idx] = _WALL

    def clear_wall(self, x, y):
        """把 ``(x, y)`` 恢复为可通行；坐标越界时抛 ``ValueError``。"""
        idx = self._index(x, y, "wall")
        with self._lock:
            self._walls[idx] = 0

    # ------------------------------------------------------------------
    # 寻路
    # ------------------------------------------------------------------
    def _point(self, point, what):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("%s must be an (x, y) pair" % what)
        x, y = point
        if type(x) is not int or type(y) is not int:
            raise ValueError("%s coordinates must be integers" % what)
        if not (0 <= x < self._w and 0 <= y < self._h):
            raise ValueError("%s coordinate out of bounds" % what)
        return x, y

    def astar(self, start, goal):
        """返回从 ``start`` 到 ``goal`` 的最短路列表；不可达时返回 ``None``。

        路径包含起点与终点，相邻点上下左右相连且不含墙。起点/终点越界或
        恰好是墙时抛 ``ValueError``。
        """
        sx, sy = self._point(start, "start")
        gx, gy = self._point(goal, "goal")

        stride = self._stride
        start_id = (sy + 1) * stride + (sx + 1)
        goal_id = (gy + 1) * stride + (gx + 1)

        # 一次性快照墙体：之后的并发 set_wall/clear_wall 只影响后续调用，
        # 本次寻路看到的是同一份一致视图，不会被撕裂。起点/终点的合法性
        # 也基于这份快照判断，因此同一次调用内不会自相矛盾。
        with self._lock:
            walls = bytes(self._walls)
            size = (self._h + 2) * stride
        if walls[start_id] or walls[goal_id]:
            raise ValueError("start or goal is a wall")
        if start_id == goal_id:
            return [(sx, sy)]

        # 开放表用按 f 值分桶的 Dial 队列（f 为整数，且单调不减），每个桶
        # 是一个栈；当前桶取空后从 fcur + 1 起扫描下一个非空桶。入队、出队
        # 都是 O(1)。桶的总数不超过格数 + w + h（f 的上界）。
        buckets = [None] * (size + self._w + self._h + 1)

        # 每个节点只会被关闭一次，出队即关闭：gs 记 0 未访问、-1 已关闭、
        # 其余存 g + 1。桶里只存节点编号。
        gs = array("i", bytes(4 * size))
        par = array("i", bytes(4 * size))

        # 启发式是 Manhattan 距离 h = |x - gx| + |y - gy|，可分离，故按行
        # 逐行构造查表；h <= w + h，通常用无符号短整型即可。Manhattan 距离
        # 满足一致性，节点第一次被弹出时 g 必为最优。
        typecode = "H" if self._w + self._h <= 0xFFFF else "I"
        heur = array(typecode, bytes(array(typecode).itemsize * size))
        row = array(typecode, (abs(x - gx) for x in range(-1, self._w + 1)))
        for yy in range(-1, self._h + 1):
            dy = abs(yy - gy)
            heur[(yy + 1) * stride:(yy + 2) * stride] = (
                row if dy == 0 else array(typecode, (c + dy for c in row))
            )

        fcur = heur[start_id]
        buckets[fcur] = [start_id]
        gs[start_id] = 1
        last_f = size + self._w + self._h

        while fcur <= last_f:
            bucket = buckets[fcur]
            if not bucket:
                fcur += 1
                continue
            while bucket:
                node = bucket.pop()
                g1 = gs[node]
                if g1 == _CLOSED:
                    continue  # 同一节点的重复条目
                if node == goal_id:
                    path = []
                    n = node
                    while n != start_id:
                        y, x = divmod(n, stride)
                        path.append((x - 1, y - 1))
                        n = par[n]
                    path.append((sx, sy))
                    path.reverse()
                    return path

                gs[node] = _CLOSED
                ng1 = g1 + 1

                # 四个方向展开写，省去循环与元组解包的开销（热路径）。
                nb = node - 1
                if not walls[nb]:
                    old = gs[nb]
                    if old != _CLOSED and (not old or old > ng1):
                        gs[nb] = ng1
                        par[nb] = node
                        f_nb = ng1 - 1 + heur[nb]
                        other = buckets[f_nb]
                        if other is None:
                            buckets[f_nb] = [nb]
                        else:
                            other.append(nb)
                nb = node + 1
                if not walls[nb]:
                    old = gs[nb]
                    if old != _CLOSED and (not old or old > ng1):
                        gs[nb] = ng1
                        par[nb] = node
                        f_nb = ng1 - 1 + heur[nb]
                        other = buckets[f_nb]
                        if other is None:
                            buckets[f_nb] = [nb]
                        else:
                            other.append(nb)
                nb = node - stride
                if not walls[nb]:
                    old = gs[nb]
                    if old != _CLOSED and (not old or old > ng1):
                        gs[nb] = ng1
                        par[nb] = node
                        f_nb = ng1 - 1 + heur[nb]
                        other = buckets[f_nb]
                        if other is None:
                            buckets[f_nb] = [nb]
                        else:
                            other.append(nb)
                nb = node + stride
                if not walls[nb]:
                    old = gs[nb]
                    if old != _CLOSED and (not old or old > ng1):
                        gs[nb] = ng1
                        par[nb] = node
                        f_nb = ng1 - 1 + heur[nb]
                        other = buckets[f_nb]
                        if other is None:
                            buckets[f_nb] = [nb]
                        else:
                            other.append(nb)
