"""A* 的边界与规模约束测试（与既有 test_core.py 互补，不修改其内容）。"""

import threading
import time
from collections import deque

import pytest

import tilekit


def _bfs_len(grid, w, h, start, goal):
    """用 BFS 求最短路步数，作为 A* 的参照。"""
    if start == goal:
        return 1
    seen = {start}
    queue = deque([(start, 1)])
    while queue:
        (x, y), dist = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            nxt = (nx, ny)
            if not (0 <= nx < w and 0 <= ny < h) or nxt in seen:
                continue
            if grid._walls[(ny + 1) * grid._stride + nx + 1]:
                continue
            if nxt == goal:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def test_astar_matches_bfs_on_obstacles():
    import random

    rng = random.Random(20261121)
    for _ in range(40):
        w, h = rng.randint(1, 14), rng.randint(1, 14)
        grid = tilekit.Grid(w, h)
        for y in range(h):
            for x in range(w):
                if rng.random() < 0.3:
                    grid.set_wall(x, y)
        for _ in range(8):
            start = (rng.randrange(w), rng.randrange(h))
            goal = (rng.randrange(w), rng.randrange(h))
            blocked = lambda p: grid._walls[(p[1] + 1) * grid._stride + p[0] + 1]
            if blocked(start) or blocked(goal):
                continue
            path = grid.astar(start, goal)
            expected = _bfs_len(grid, w, h, start, goal)
            assert (path is None) == (expected is None)
            if path is not None:
                assert len(path) == expected


def test_astar_diagonal_not_used():
    # 对角移动若被允许，长度会是 3；四邻域最短路必须经过一个拐点。
    path = tilekit.Grid(3, 3).astar((0, 0), (2, 2))
    assert len(path) == 5
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)] or (
        path[0] == (0, 0) and path[-1] == (2, 2)
    )
    assert all(
        max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(path, path[1:])
    )


def test_astar_start_equals_goal_and_wall():
    grid = tilekit.Grid(4, 4)
    assert grid.astar((2, 3), (2, 3)) == [(2, 3)]
    grid.set_wall(2, 3)
    with pytest.raises(ValueError):
        grid.astar((2, 3), (2, 3))


def test_astar_boundary_out_of_range():
    grid = tilekit.Grid(4, 4)
    for point in ((-1, 0), (0, -1), (4, 0), (0, 4), (4, 4)):
        with pytest.raises(ValueError):
            grid.astar(point, (1, 1))
        with pytest.raises(ValueError):
            grid.astar((1, 1), point)


def test_astar_large_grid_is_fast():
    n = 256
    grid = tilekit.Grid(n, n)
    started = time.perf_counter()
    path = grid.astar((0, 0), (n - 1, n - 1))
    elapsed = time.perf_counter() - started
    assert path is not None and len(path) == 2 * n - 1
    assert elapsed < 0.2, "256x256 用时 %.3f s" % elapsed


def test_astar_concurrent_same_grid_is_deterministic():
    grid = tilekit.Grid(64, 64)
    for i in range(0, 64, 5):
        grid.set_wall(i, (i * 3) % 64)
    grid.clear_wall(0, 0)
    reference = grid.astar((0, 0), (63, 63))
    results = []
    lock = threading.Lock()
    errors = []

    def worker():
        try:
            path = grid.astar((0, 0), (63, 63))
        except Exception as exc:  # pragma: no cover - 失败时用于报告
            errors.append(exc)
            return
        with lock:
            results.append(path)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert results and all(path == reference for path in results)
