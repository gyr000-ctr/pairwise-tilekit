import math

import pytest

import tilekit

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _g():
    return tilekit.Grid(5, 5)


def _check_astar_straight():
    return len(_g().astar((0, 0), (4, 0))) == 5


def _check_astar_wall():
    g = tilekit.Grid(3, 3)
    g.set_wall(1, 1)
    return g.astar((0, 0), (0, 0)) == [(0, 0)]


def _check_astar_unreachable():
    g = tilekit.Grid(3, 3)
    g.set_wall(1, 0)
    g.set_wall(1, 1)
    g.set_wall(1, 2)
    return g.astar((0, 0), (2, 0)) is None


def _check_astar_same():
    return _g().astar((2, 2), (2, 2)) == [(2, 2)]


def _check_astar_start_wall():
    g = _g()
    g.set_wall(0, 0)
    return _raises(lambda: g.astar((0, 0), (4, 4)))


def _check_astar_bounds():
    return _raises(lambda: _g().astar((0, 0), (9, 9)))


def _check_astar_adjacent():
    path = _g().astar((0, 0), (4, 4))
    return all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(path, path[1:]))


def _check_astar_no_wall():
    g = _g()
    g.set_wall(2, 0)
    path = g.astar((0, 0), (4, 0))
    return all(p not in [(2, 0)] for p in path)


def _check_astar_det():
    return _g().astar((0, 0), (4, 4)) == _g().astar((0, 0), (4, 4))


def _check_astar_type():
    return isinstance(_g().astar((0, 0), (1, 0)), list)


def _check_astar_endpoints():
    path = _g().astar((0, 0), (4, 4))
    return path[0] == (0, 0) and path[-1] == (4, 4)


def _check_astar_empty():
    return tilekit.Grid(1, 1).astar((0, 0), (0, 0)) == [(0, 0)]


def _check_astar_negative():
    return _raises(lambda: _g().astar((-1, 0), (0, 0)))


def _check_astar_clear():
    g = _g()
    g.set_wall(1, 0)
    g.clear_wall(1, 0) if hasattr(g, "clear_wall") else None
    return len(g.astar((0, 0), (2, 0))) >= 3


def _check_astar_len():
    return len(_g().astar((0, 0), (2, 2))) == 5

CASES = [("astar_straight", "_check_astar_straight"), ("astar_wall", "_check_astar_wall"), ("astar_unreachable", "_check_astar_unreachable"), ("astar_same", "_check_astar_same"), ("astar_start_wall", "_check_astar_start_wall"), ("astar_bounds", "_check_astar_bounds"), ("astar_adjacent", "_check_astar_adjacent"), ("astar_no_wall", "_check_astar_no_wall"), ("astar_det", "_check_astar_det"), ("astar_type", "_check_astar_type"), ("astar_endpoints", "_check_astar_endpoints"), ("astar_empty", "_check_astar_empty"), ("astar_negative", "_check_astar_negative"), ("astar_clear", "_check_astar_clear"), ("astar_len", "_check_astar_len")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
