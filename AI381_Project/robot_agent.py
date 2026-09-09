# file: robot_agent.py
# Robot with BFS, DFS, Greedy, and A* (animated-friendly planners).
# Public methods:
#   - find_object_animated(viewer, delay_ms, method="astar"|"bfs"|"dfs"|"greedy")
#   - move_to_target_animated(viewer, delay_ms, method="astar"|"bfs"|"dfs"|"greedy")
#   - return_to_start_animated(viewer, delay_ms, method="astar"|"bfs"|"dfs"|"greedy")
#
# Notes:
# - The robot talks to MapServer only via getRoom() and getSize().
# - Coordinates inside planners use (row, col). MapServer stores S/O/T as (x=col, y=row).
# - 'viewer' is any object exposing redraw_step(visited, path, delay_ms); pass a dummy for headless runs.

from collections import deque
import heapq
from typing import Dict, List, Optional, Tuple

# Must match MapServer’s direction order everywhere
RIGHT, TOP, LEFT, DOWN = 0, 1, 2, 3


def reconstruct_path(parent: Dict[Tuple[int, int], Tuple[int, int]],
                     start: Tuple[int, int],
                     goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Rebuild path by following parent pointers from goal→start (then reverse)."""
    path: List[Tuple[int, int]] = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path


class Robot:
    def __init__(self, mapserver) -> None:
        """Capture map handle, convert S/O/T into (row, col), init logs and state."""
        self.map = mapserver

        # MapServer: start is (x=col, y=row) → convert to (row, col)
        sx, sy = self.map.start
        self.start_rc = (sy, sx)
        self.cur_pos = self.start_rc

        # Object/Target might be None when unknown; convert if present
        self.object_rc = (self.map.object[1], self.map.object[0]) if getattr(self.map, "object", None) else None
        self.target_rc = (self.map.target[1], self.map.target[0]) if getattr(self.map, "target", None) else None

        # Carry state + traces
        self.carrying_object: int = 0
        self.visited_nodes: List[Tuple[int, int]] = []     # union of expanded nodes across all phases
        self.path_history: List[Tuple[int, int]] = [self.start_rc]  # concatenated paths

        self.rows, self.cols = self.map.getSize()

    # ---------- utilities ----------

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Adjacency respecting walls: return valid (row, col) moves in 4-neighborhood."""
        walls = self.map.getRoom(r, c)["walls"]
        out: List[Tuple[int, int]] = []
        if walls[RIGHT] == 0 and c + 1 < self.cols:
            out.append((r, c + 1))
        if walls[TOP] == 0 and r - 1 >= 0:
            out.append((r - 1, c))
        if walls[LEFT] == 0 and c - 1 >= 0:
            out.append((r, c - 1))
        if walls[DOWN] == 0 and r + 1 < self.rows:
            out.append((r + 1, c))
        return out

    def _append_expanded(self, expanded: List[Tuple[int, int]]) -> None:
        """Merge newly expanded nodes into the global visited log (deduplicated)."""
        seen = set(self.visited_nodes)
        for node in expanded:
            if node not in seen:
                self.visited_nodes.append(node)
                seen.add(node)

    def _append_path_to_history(self, path: List[Tuple[int, int]]) -> None:
        """Concatenate path to history; avoid duplicating the join vertex."""
        if not path:
            return
        if self.path_history and self.path_history[-1] == path[0]:
            self.path_history.extend(path[1:])
        else:
            self.path_history.extend(path)

    # ---------- planners (with optional viewer animation) ----------

    def _bfs_with_animation(
        self,
        start: Tuple[int, int],
        goal: Optional[Tuple[int, int]],
        viewer, delay_ms: int, goal_test=None,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Breadth-First Search; optimal on uniform-cost grids."""
        q = deque([start])
        seen = {start}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        expanded: List[Tuple[int, int]] = []

        def is_goal(rc: Tuple[int, int]) -> bool:
            return rc == goal if goal is not None else goal_test(*rc)

        while q:
            cur = q.popleft()
            expanded.append(cur)
            if viewer: viewer.redraw_step(visited=expanded, path=[], delay_ms=delay_ms)
            if is_goal(cur):
                path = reconstruct_path(parent, start, cur)
                if viewer: viewer.redraw_step(visited=expanded, path=path, delay_ms=delay_ms)
                return path, expanded
            for n in self._neighbors(*cur):
                if n not in seen:
                    seen.add(n)
                    parent[n] = cur
                    q.append(n)
        if viewer: viewer.redraw_step(visited=expanded, path=[], delay_ms=delay_ms)
        return [], expanded

    def _dfs_with_animation(
        self,
        start: Tuple[int, int],
        goal: Optional[Tuple[int, int]],
        viewer, delay_ms: int, goal_test=None,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Depth-First Search; not optimal but simple and often fast in narrow mazes."""
        stack = [start]
        seen = {start}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        expanded: List[Tuple[int, int]] = []

        def is_goal(rc: Tuple[int, int]) -> bool:
            return rc == goal if goal is not None else goal_test(*rc)

        while stack:
            cur = stack.pop()
            expanded.append(cur)
            if viewer: viewer.redraw_step(visited=expanded, path=[], delay_ms=delay_ms)
            if is_goal(cur):
                path = reconstruct_path(parent, start, cur)
                if viewer: viewer.redraw_step(visited=expanded, path=path, delay_ms=delay_ms)
                return path, expanded
            # reversed() → explore neighbors in a stable order when pushed
            for n in reversed(self._neighbors(*cur)):
                if n not in seen:
                    seen.add(n)
                    parent[n] = cur
                    stack.append(n)
        if viewer: viewer.redraw_step(visited=expanded, path=[], delay_ms=delay_ms)
        return [], expanded

    def _greedy_with_animation(
        self,
        start: Tuple[int, int],
        goal: Optional[Tuple[int, int]],
        viewer, delay_ms: int, goal_test=None,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Greedy Best-First: expand the frontier node with the smallest heuristic.
        Fast, but not optimal.

        Requires a CONCRETE goal cell. With `goal=None` (predicate goal test,
        e.g. searching for an object whose position is unknown) the Manhattan
        heuristic has nothing to measure against, so it would return 0 for every
        node and Greedy would silently degenerate into a DFS over the heap.
        That is misleading in a comparison, so it is rejected outright.

        Raises:
            ValueError: if `goal` is None.
        """
        if goal is None:
            raise ValueError(
                "greedy best-first search requires a known goal cell; "
                "it cannot run with a predicate-only goal test because the "
                "heuristic would be identically 0 (degenerating into DFS). "
                "Use 'bfs', 'dfs' or 'astar' when the goal position is unknown."
            )

        # Manhattan heuristic towards the known goal
        def h(a: Tuple[int, int]) -> int:
            return abs(a[0] - goal[0]) + abs(a[1] - goal[1])

        def is_goal(rc: Tuple[int, int]) -> bool:
            return rc == goal if goal is not None else goal_test(*rc)

        open_heap: List[Tuple[int, Tuple[int, int]]] = [(h(start), start)]
        seen: set = {start}  # avoid re-adding
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        expanded: List[Tuple[int, int]] = []

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            expanded.append(cur)
            if viewer: viewer.redraw_step(visited=expanded, path=[], delay_ms=delay_ms)
            if is_goal(cur):
                path = reconstruct_path(parent, start, cur)
                if viewer: viewer.redraw_step(visited=expanded, path=path, delay_ms=delay_ms)
                return path, expanded
            for n in self._neighbors(*cur):
                if n in seen:
                    continue
                seen.add(n)
                parent[n] = cur
                heapq.heappush(open_heap, (h(n), n))
        if viewer: viewer.redraw_step(visited=expanded, path=[], delay_ms=delay_ms)
        return [], expanded

    def _astar_with_animation(
        self,
        start: Tuple[int, int],
        goal: Optional[Tuple[int, int]],
        viewer, delay_ms: int, goal_test=None,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        A* with Manhattan heuristic; optimal with non-negative uniform costs.

        With `goal=None` (predicate goal test) the heuristic is 0 everywhere,
        which makes this a uniform-cost search. That is still optimal, so unlike
        Greedy it remains a meaningful search and is allowed.
        """
        def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def is_goal(rc: Tuple[int, int]) -> bool:
            return rc == goal if goal is not None else goal_test(*rc)

        open_heap: List[Tuple[int, Tuple[int, int]]] = [(0, start)]  # (f, node)
        g: Dict[Tuple[int, int], int] = {start: 0}                    # cost-so-far
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed: set = set()
        expanded: List[Tuple[int, int]] = []

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            expanded.append(cur)
            if viewer: viewer.redraw_step(visited=expanded, path=[], delay_ms=delay_ms)

            if is_goal(cur):
                path = reconstruct_path(parent, start, cur)
                if viewer: viewer.redraw_step(visited=expanded, path=path, delay_ms=delay_ms)
                return path, expanded

            for n in self._neighbors(*cur):
                step_cost = self.map.getRoom(*n)["cost"]          # supports weighted cells later
                tentative = g[cur] + step_cost
                if tentative < g.get(n, 1 << 60):                 # better path found
                    g[n] = tentative
                    parent[n] = cur
                    h = 0 if goal is None else manhattan(n, goal) # 0 when goal test is predicate
                    heapq.heappush(open_heap, (tentative + h, n))

        if viewer: viewer.redraw_step(visited=expanded, path=[], delay_ms=delay_ms)
        return [], expanded

    # ---------- dispatch ----------

    # Name -> planner method. Used by _plan(); keeps the four public phases free
    # of repeated if/elif chains and makes an unknown method an explicit error.
    _PLANNERS = {
        "bfs": _bfs_with_animation,
        "dfs": _dfs_with_animation,
        "greedy": _greedy_with_animation,
        "astar": _astar_with_animation,
    }

    def _plan(
        self,
        start: Tuple[int, int],
        goal: Optional[Tuple[int, int]],
        viewer,
        delay_ms: int,
        method: str,
        goal_test=None,
    ) -> List[Tuple[int, int]]:
        """
        Run one planner and fold its result into the robot's traces.

        Args:
            start: (row, col) to plan from.
            goal: (row, col) destination, or None when `goal_test` decides.
            viewer: object with redraw_step(visited, path, delay_ms), or None.
            delay_ms: per-frame animation delay.
            method: one of 'bfs', 'dfs', 'greedy', 'astar' (case-insensitive).
            goal_test: predicate goal_test(row, col) -> bool, used when goal is None.

        Returns:
            The planned path as a list of (row, col), or [] if unreachable.

        Raises:
            ValueError: if `method` is not a known planner name.
        """
        key = str(method).lower()
        planner = self._PLANNERS.get(key)
        if planner is None:
            raise ValueError(
                f"unknown search method {method!r}; "
                f"expected one of {sorted(self._PLANNERS)}"
            )

        path, expanded = planner(self, start, goal, viewer, delay_ms, goal_test)

        # Bookkeeping shared by every phase
        self._append_expanded(expanded)
        if path:
            self._append_path_to_history(path)
            self.cur_pos = path[-1]
        return path

    # ---------- public phases ----------

    def find_object_animated(self, viewer, delay_ms: int, method: str = "astar") -> List[Tuple[int, int]]:
        """Plan from current position to the object cell. Unknown object -> predicate goal test."""
        if self.object_rc is None:
            goal_test = lambda r, c: self.map.getRoom(r, c)["obj"] == 1
            goal = None
        else:
            goal_test = None
            goal = self.object_rc

        path = self._plan(self.cur_pos, goal, viewer, delay_ms, method, goal_test)
        if path:
            self.carrying_object = 1
            if self.object_rc is None:
                self.object_rc = path[-1]  # now known
        return path

    def move_to_target_animated(self, viewer, delay_ms: int, method: str = "astar") -> List[Tuple[int, int]]:
        """Plan from current position (carrying=1) to target cell; clears carrying on arrival."""
        if self.carrying_object != 1:
            raise RuntimeError("Collect the object first.")
        if self.target_rc is None:
            raise RuntimeError("Target must be known.")

        path = self._plan(self.cur_pos, self.target_rc, viewer, delay_ms, method)
        if path:
            self.carrying_object = 0
        return path

    def return_to_start_animated(self, viewer, delay_ms: int, method: str = "astar") -> List[Tuple[int, int]]:
        """Plan from current position back to the start cell."""
        return self._plan(self.cur_pos, self.start_rc, viewer, delay_ms, method)
