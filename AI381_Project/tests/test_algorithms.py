"""
Tests for the maze generator and the four search algorithms.

Run from the project root with:  pytest -q
"""
import json
import os
import sys

import pytest

from mapserver import MapServer, RIGHT, TOP, LEFT, DOWN
from robot_agent import Robot

# Seeds exercised by the parametrised tests. Enough to catch layout-specific luck.
SEEDS = list(range(1, 21))

ROWS, COLS = 12, 16
BRAID = 0.15

# (dr, dc) -> direction index, matching the project-wide [RIGHT, TOP, LEFT, DOWN] order
_STEP_TO_DIR = {(0, 1): RIGHT, (-1, 0): TOP, (0, -1): LEFT, (1, 0): DOWN}

ALGORITHMS = ("bfs", "dfs", "greedy", "astar")


class DummyViewer:
    """No-op viewer so planners can run headless."""

    def redraw_step(self, visited, path, delay_ms):
        return


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_maze(seed, braid_factor=BRAID, rows=ROWS, cols=COLS):
    """Build one braided maze with a fixed seed."""
    m = MapServer(rows=rows, cols=cols)
    m.generate_maze(seed=seed, braid_factor=braid_factor)
    return m


def run_all_phases(m, method):
    """Run find-object -> move-to-target -> return-to-start and return the robot."""
    robot = Robot(m)
    viewer = DummyViewer()
    robot.find_object_animated(viewer, delay_ms=0, method=method)
    robot.move_to_target_animated(viewer, delay_ms=0, method=method)
    robot.return_to_start_animated(viewer, delay_ms=0, method=method)
    return robot


def path_length(m, method):
    """Total number of moves made across all three phases."""
    return len(run_all_phases(m, method).path_history) - 1


def assert_path_is_walkable(m, path):
    """
    A path must be a chain of orthogonally adjacent cells with no wall between
    consecutive cells, and must stay inside the grid.
    """
    rows, cols = m.getSize()
    assert path, "path must not be empty"

    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols, f"cell {(r, c)} is outside the grid"

    for (r, c), (nr, nc) in zip(path, path[1:]):
        step = (nr - r, nc - c)
        assert step in _STEP_TO_DIR, (
            f"step from {(r, c)} to {(nr, nc)} is not a single orthogonal move"
        )
        d = _STEP_TO_DIR[step]
        walls_here = m.getRoom(r, c)["walls"]
        walls_there = m.getRoom(nr, nc)["walls"]
        opposite = {RIGHT: LEFT, LEFT: RIGHT, TOP: DOWN, DOWN: TOP}[d]
        assert walls_here[d] == 0, f"path crosses a wall leaving {(r, c)} towards {(nr, nc)}"
        assert walls_there[opposite] == 0, (
            f"reciprocal wall still standing between {(nr, nc)} and {(r, c)}"
        )


def open_passage_count(m):
    """Number of open passages, counting each shared wall segment once."""
    total = 0
    for r in range(m.rows):
        for c in range(m.cols):
            for d in (RIGHT, DOWN):
                dr, dc = m._delta(d)
                nr, nc = r + dr, c + dc
                if m._in_bounds(nr, nc) and m.maze[r][c]["walls"][d] == 0:
                    total += 1
    return total


# --------------------------------------------------------------------------
# optimality: BFS and A* must agree
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_bfs_and_astar_return_equal_path_lengths(seed):
    """Both are optimal on a uniform-cost grid, so their path lengths must match."""
    m = make_maze(seed)
    assert path_length(m, "bfs") == path_length(m, "astar")


@pytest.mark.parametrize("seed", SEEDS)
def test_dfs_is_never_shorter_than_bfs(seed):
    """DFS is not optimal, so it can only tie or lose against BFS."""
    m = make_maze(seed)
    assert path_length(m, "dfs") >= path_length(m, "bfs")


@pytest.mark.parametrize("seed", SEEDS)
def test_greedy_is_never_shorter_than_bfs(seed):
    """Greedy best-first is not optimal either."""
    m = make_maze(seed)
    assert path_length(m, "greedy") >= path_length(m, "bfs")


def test_braiding_makes_the_comparison_meaningful():
    """
    The point of braiding: on a perfect maze every algorithm returns the SAME
    path, so the comparison says nothing. With loops present, DFS should be
    strictly worse than BFS on the large majority of seeds.
    """
    strictly_longer = sum(
        1 for seed in SEEDS
        if path_length(make_maze(seed), "dfs") > path_length(make_maze(seed), "bfs")
    )
    assert strictly_longer >= 0.75 * len(SEEDS), (
        f"DFS was strictly longer on only {strictly_longer}/{len(SEEDS)} seeds; "
        "braiding does not appear to be introducing loops"
    )


def test_perfect_maze_makes_all_algorithms_identical():
    """Regression guard documenting the original bug (braid_factor=0.0)."""
    m = make_maze(seed=5, braid_factor=0.0)
    lengths = {alg: path_length(m, alg) for alg in ALGORITHMS}
    assert len(set(lengths.values())) == 1, lengths


# --------------------------------------------------------------------------
# path validity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS[:10])
@pytest.mark.parametrize("method", ALGORITHMS)
def test_paths_are_contiguous_and_never_cross_a_wall(seed, method):
    """Every phase path, and the concatenation of all three, must be walkable."""
    m = make_maze(seed)
    robot = Robot(m)
    viewer = DummyViewer()

    p1 = robot.find_object_animated(viewer, delay_ms=0, method=method)
    p2 = robot.move_to_target_animated(viewer, delay_ms=0, method=method)
    p3 = robot.return_to_start_animated(viewer, delay_ms=0, method=method)

    for phase in (p1, p2, p3):
        assert_path_is_walkable(m, phase)
    assert_path_is_walkable(m, robot.path_history)


@pytest.mark.parametrize("seed", SEEDS[:10])
@pytest.mark.parametrize("method", ALGORITHMS)
def test_paths_start_and_end_where_expected(seed, method):
    """start -> object -> target -> start."""
    m = make_maze(seed)
    robot = Robot(m)
    viewer = DummyViewer()
    start_rc = robot.start_rc
    object_rc = robot.object_rc
    target_rc = robot.target_rc

    p1 = robot.find_object_animated(viewer, delay_ms=0, method=method)
    p2 = robot.move_to_target_animated(viewer, delay_ms=0, method=method)
    p3 = robot.return_to_start_animated(viewer, delay_ms=0, method=method)

    assert p1[0] == start_rc and p1[-1] == object_rc
    assert p2[0] == object_rc and p2[-1] == target_rc
    assert p3[0] == target_rc and p3[-1] == start_rc


# --------------------------------------------------------------------------
# maze generation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS[:5])
def test_braid_factor_zero_produces_a_perfect_maze(seed):
    """A perfect maze on N cells is a spanning tree: exactly N-1 open passages."""
    m = make_maze(seed, braid_factor=0.0)
    assert open_passage_count(m) == m.rows * m.cols - 1


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_braiding_adds_passages(seed):
    """Braiding must open strictly more passages than a spanning tree has."""
    m = make_maze(seed, braid_factor=BRAID)
    assert open_passage_count(m) > m.rows * m.cols - 1


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_every_cell_is_reachable(seed):
    """Carving must connect the whole grid; braiding can only add connections."""
    m = make_maze(seed)
    robot = Robot(m)
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        cur = stack.pop()
        for n in robot._neighbors(*cur):
            if n not in seen:
                seen.add(n)
                stack.append(n)
    assert len(seen) == m.rows * m.cols


def test_generation_is_deterministic_for_a_seed():
    a = make_maze(seed=42)
    b = make_maze(seed=42)
    assert a.maze == b.maze
    assert (a.start, a.object, a.target) == (b.start, b.object, b.target)


def test_large_maze_does_not_hit_the_recursion_limit():
    """
    The old recursive carver raised RecursionError above roughly 110x110.
    The iterative carver must handle a grid far past that.
    """
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)  # far below the number of cells
    try:
        m = MapServer(rows=150, cols=150)
        m.generate_maze(seed=1, braid_factor=0.0)
    finally:
        sys.setrecursionlimit(limit)
    assert open_passage_count(m) == 150 * 150 - 1


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_braid_factor_is_rejected(bad):
    with pytest.raises(ValueError):
        MapServer(rows=5, cols=5).generate_maze(seed=1, braid_factor=bad)


# --------------------------------------------------------------------------
# JSON round-trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS[:5])
def test_save_load_round_trips_to_an_identical_grid(seed, tmp_path):
    original = make_maze(seed)
    path = tmp_path / "maze.json"
    original.SaveMaze(str(path))

    reloaded = MapServer(rows=1, cols=1)  # deliberately wrong size; JSON overrides it
    reloaded.LoadMaze(str(path))

    assert (reloaded.rows, reloaded.cols) == (original.rows, original.cols)
    assert reloaded.start == original.start
    assert reloaded.object == original.object
    assert reloaded.target == original.target
    assert reloaded.maze == original.maze

    for r in range(original.rows):
        for c in range(original.cols):
            assert reloaded.getRoom(r, c) == original.getRoom(r, c)


def test_save_load_preserves_search_results(tmp_path):
    """The reloaded maze must give byte-identical planning results."""
    original = make_maze(seed=9)
    path = tmp_path / "maze.json"
    original.SaveMaze(str(path))
    reloaded = MapServer(rows=1, cols=1)
    reloaded.LoadMaze(str(path))

    for method in ALGORITHMS:
        assert (run_all_phases(original, method).path_history
                == run_all_phases(reloaded, method).path_history)


def test_saved_json_is_valid_and_row_major(tmp_path):
    m = make_maze(seed=3)
    path = tmp_path / "maze.json"
    m.SaveMaze(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["rows"] == m.rows and data["cols"] == m.cols
    assert len(data["maze"]) == m.rows * m.cols
    assert data["maze"][0]["coords"] == [0, 0]
    assert data["maze"][m.cols]["coords"] == [1, 0]  # row-major ordering


# --------------------------------------------------------------------------
# dispatch and error handling
# --------------------------------------------------------------------------

def test_unknown_method_raises_value_error():
    """An unknown planner name must fail loudly, not silently fall back to A*."""
    robot = Robot(make_maze(seed=1))
    with pytest.raises(ValueError, match="unknown search method"):
        robot.find_object_animated(DummyViewer(), delay_ms=0, method="dijkstra")


@pytest.mark.parametrize("method", ALGORITHMS)
def test_method_name_is_case_insensitive(method):
    m = make_maze(seed=1)
    robot = Robot(m)
    path = robot.find_object_animated(DummyViewer(), delay_ms=0, method=method.upper())
    assert path[-1] == robot.object_rc


def test_greedy_requires_a_known_goal():
    """
    With goal=None the Manhattan heuristic is 0 everywhere and Greedy would
    silently degenerate into DFS, so it must refuse instead.
    """
    robot = Robot(make_maze(seed=1))
    with pytest.raises(ValueError, match="requires a known goal"):
        robot._plan(robot.start_rc, None, DummyViewer(), 0, "greedy",
                    goal_test=lambda r, c: False)


def test_astar_still_works_with_a_predicate_goal():
    """Unlike Greedy, A* degrades to a correct uniform-cost search when goal is None."""
    m = make_maze(seed=1)
    robot = Robot(m)
    robot.object_rc = None  # force the predicate branch
    path = robot.find_object_animated(DummyViewer(), delay_ms=0, method="astar")
    assert m.getRoom(*path[-1])["obj"] == 1


def test_move_to_target_requires_the_object_first():
    robot = Robot(make_maze(seed=1))
    with pytest.raises(RuntimeError, match="Collect the object first"):
        robot.move_to_target_animated(DummyViewer(), delay_ms=0, method="bfs")
