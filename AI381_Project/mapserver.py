# file: mapserver.py
# MapServer for AI381 project.
#
# Cell format used INTERNALLY for each grid position (row, col):
# {
#   "walls": [right, top, left, down],   # 1 = wall present, 0 = open passage
#   "obj": 0 | 1 | -1,                   # 0 = normal, 1 = object cell, -1 = target cell
#   "coords": [row, col],                # stored for convenience/debugging
#   "cost": 1                            # movement cost (currently uniform)
# }
#
# JSON format produced/consumed by SaveMaze/LoadMaze in THIS VERSION:
# {
#   "rows": 10,
#   "cols": 10,
#   "start": [x, y],                     # (x = column, y = row)
#   "object": [x, y],
#   "target": [x, y],
#   "maze": [
#      { "walls": [...], "obj": 0, "coords": [r,c], "cost": 1 },  # flat list (row-major)
#      ...
#   ]
# }
# NOTE: LoadMaze also accepts a 2D "maze" [[{...},...],...] for robustness.
#       Coordinates in the API always use (x, y) = (col, row), while the grid is indexed (row, col).

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import json
import random

# Direction indices shared across the project.
# IMPORTANT: The order must remain [Right, Top, Left, Down] everywhere.
RIGHT, TOP, LEFT, DOWN = 0, 1, 2, 3

# Opposite-direction lookup used for carving/adding reciprocal walls.
_OPP = {RIGHT: LEFT, LEFT: RIGHT, TOP: DOWN, DOWN: TOP}


@dataclass
class _Cell:
    """
    Lightweight container for a single maze cell before converting to a dict.
    Using a dataclass keeps initialization clear; we immediately convert to a dict
    to match the required internal format.
    """
    walls: List[int]          # [R, T, L, D] with 1 = wall, 0 = open
    obj: int                  # 0 = normal, 1 = object, -1 = target
    coords: List[int]         # [row, col] (for reference)
    cost: int = 1             # movement cost (uniform for Phase 1–4)

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow-copied dict matching the internal cell schema."""
        return {
            "walls": self.walls[:],
            "obj": self.obj,
            "coords": self.coords[:],
            "cost": self.cost,
        }


class MapServer:
    """
    Stores and manages the maze grid plus Start/Object/Target positions.

    Coordinate convention:
      - Public positions are stored as tuples (x, y) where x = column, y = row.
      - Internal grid access is by (row, col) → self.maze[row][col].

    Public attributes:
      - self.rows, self.cols: grid dimensions
      - self.maze: 2D list of cell dicts (see header comment for schema)
      - self.start:  (x, y) or None
      - self.object: (x, y) or None
      - self.target: (x, y) or None
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        start_x: Optional[int] = None,
        start_y: Optional[int] = None,
        object_x: Optional[int] = None,
        object_y: Optional[int] = None,
        target_x: Optional[int] = None,
        target_y: Optional[int] = None,
    ) -> None:
        # Store dimensions
        self.rows = int(rows)
        self.cols = int(cols)

        # Initialize a fully closed grid (all walls = 1)
        self.maze: List[List[Dict[str, Any]]] = [
            [self._new_cell(r, c) for c in range(self.cols)] for r in range(self.rows)
        ]

        # Entities are stored as (x, y) = (col, row); may be None until placed.
        self.start: Optional[Tuple[int, int]] = (
            (start_x, start_y) if start_x is not None and start_y is not None else None
        )
        self.object: Optional[Tuple[int, int]] = (
            (object_x, object_y) if object_x is not None and object_y is not None else None
        )
        self.target: Optional[Tuple[int, int]] = (
            (target_x, target_y) if target_x is not None and target_y is not None else None
        )

    # -------------------- Cell helpers --------------------

    def _new_cell(self, r: int, c: int) -> Dict[str, Any]:
        """
        Create a new cell with all walls present.
        r,c are internal indices (row, col).
        """
        return _Cell(walls=[1, 1, 1, 1], obj=0, coords=[r, c], cost=1).to_dict()

    def _in_bounds(self, r: int, c: int) -> bool:
        """Return True if (r, c) is inside the grid."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    # -------------------- Public API --------------------

    def getRoom(self, row: int, col: int) -> Dict[str, Any]:
        """
        Return a COPY of a single cell in the required format.
        The copy prevents external mutation of internal grid state.
        """
        cell = self.maze[row][col]
        return {
            "walls": cell["walls"][:],
            "obj": cell["obj"],             # internal key used by later phases
            "coords": cell["coords"][:],    # provided for debugging/consistency
            "cost": cell["cost"],
        }

    def getSize(self) -> Tuple[int, int]:
        """Return (rows, cols)."""
        return self.rows, self.cols

    # Optional per spec
    def setSize(self, rows: int, cols: int) -> None:
        """
        Resize the grid and reset all cells to closed walls/cost=1.
        NOTE: Start/Object/Target are reset to None; caller must reassign.
        """
        self.rows = int(rows)
        self.cols = int(cols)
        self.maze = [[self._new_cell(r, c) for c in range(self.cols)] for r in range(self.rows)]
        self.start = None
        self.object = None
        self.target = None

    # Optional per spec
    def addWall(self, row: int, col: int, direction: int) -> None:
        """
        Add a wall to cell (row, col) in 'direction', and add the reciprocal wall
        in the adjacent neighbor cell if it exists.
        direction must be one of (RIGHT, TOP, LEFT, DOWN).
        """
        if direction not in (RIGHT, TOP, LEFT, DOWN):
            raise ValueError("direction must be 0..3 for [Right, Top, Left, Down]")
        if not self._in_bounds(row, col):
            raise IndexError("cell out of bounds")

        # Add wall at current cell
        self.maze[row][col]["walls"][direction] = 1

        # Add opposite wall at neighbor (if inside grid)
        dr, dc = self._delta(direction)
        nr, nc = row + dr, col + dc
        if self._in_bounds(nr, nc):
            self.maze[nr][nc]["walls"][_OPP[direction]] = 1

    # -------------------- Maze generation / placement --------------------

    def generate_maze(self, seed: Optional[int] = None, braid_factor: float = 0.15) -> None:
        """
        Generate a maze using a randomized depth-first "recursive backtracker",
        then optionally BRAID it by knocking out some of the remaining walls.

        Args:
            seed: RNG seed. The same seed always yields the same maze.
            braid_factor: fraction (0.0-1.0) of the interior walls still standing
                after carving that are removed at random. 0.0 leaves a PERFECT
                (loop-free) maze; anything above 0.0 introduces loops.

        Why braiding matters:
            A perfect maze has exactly ONE path between any two cells, so BFS,
            DFS, Greedy and A* all return the identical path and the comparison
            is meaningless. Adding loops gives the algorithms real choices, so
            optimal searches (BFS/A*) can be distinguished from DFS/Greedy.

        Start/Object/Target are placed afterwards (randomly if missing).
        """
        if not 0.0 <= braid_factor <= 1.0:
            raise ValueError("braid_factor must be between 0.0 and 1.0")

        rng = random.Random(seed)

        # Reset all walls to present (1)
        for r in range(self.rows):
            for c in range(self.cols):
                self.maze[r][c]["walls"] = [1, 1, 1, 1]

        # Carve a perfect maze from a random starting cell
        self._carve_iterative(rng, rng.randrange(self.rows), rng.randrange(self.cols))

        # Knock out extra walls to create loops (no-op when braid_factor == 0)
        self._braid(rng, braid_factor)

        # Ensure entities exist and are distinct; place randomly if not provided
        self._place_entities_randomly(rng)

        # Update 'obj' flags in the grid to reflect object/target positions
        self._apply_entity_flags()

    def _carve_iterative(self, rng: random.Random, r0: int, c0: int) -> None:
        """
        Randomized DFS carving using an EXPLICIT STACK (no recursion).

        The recursive form overflowed Python's call stack on grids larger than
        roughly 110x110, because the carving path can visit every cell before
        backtracking. The stack holds (row, col, remaining_directions) frames,
        which makes the depth a heap list instead of interpreter frames.

        Produces a perfect (loop-free, fully connected) maze.
        """
        def fresh_dirs() -> List[int]:
            dirs = [RIGHT, TOP, LEFT, DOWN]
            rng.shuffle(dirs)  # randomize expansion order
            return dirs

        visited = [[False] * self.cols for _ in range(self.rows)]
        visited[r0][c0] = True
        stack: List[Tuple[int, int, List[int]]] = [(r0, c0, fresh_dirs())]

        while stack:
            r, c, dirs = stack[-1]
            advanced = False

            # Try the remaining directions of the current frame
            while dirs:
                d = dirs.pop()
                dr, dc = self._delta(d)
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc) and not visited[nr][nc]:
                    # Open passage in both cells: current(d) and neighbor(opposite d)
                    self.maze[r][c]["walls"][d] = 0
                    self.maze[nr][nc]["walls"][_OPP[d]] = 0
                    visited[nr][nc] = True
                    stack.append((nr, nc, fresh_dirs()))   # "recurse"
                    advanced = True
                    break

            if not advanced:
                stack.pop()  # dead end: backtrack

    def _braid(self, rng: random.Random, braid_factor: float) -> int:
        """
        Remove `braid_factor` of the interior walls that are still standing,
        chosen uniformly at random, opening BOTH reciprocal walls each time.

        Only RIGHT and DOWN are scanned so every interior wall segment is
        considered exactly once (the LEFT/TOP views of the same segment belong
        to the neighbouring cell).

        Returns the number of wall segments removed.
        """
        if braid_factor <= 0.0:
            return 0

        candidates: List[Tuple[int, int, int, int, int]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                for d in (RIGHT, DOWN):
                    dr, dc = self._delta(d)
                    nr, nc = r + dr, c + dc
                    if self._in_bounds(nr, nc) and self.maze[r][c]["walls"][d] == 1:
                        candidates.append((r, c, d, nr, nc))

        n_remove = min(len(candidates), int(round(braid_factor * len(candidates))))
        for (r, c, d, nr, nc) in rng.sample(candidates, n_remove):
            self.maze[r][c]["walls"][d] = 0
            self.maze[nr][nc]["walls"][_OPP[d]] = 0
        return n_remove

    def assign_objects(
        self,
        start_x: int,
        start_y: int,
        object_x: int,
        object_y: int,
        target_x: int,
        target_y: int,
    ) -> None:
        """
        Reposition Start/Object/Target using (x, y) = (col, row) coordinates
        and update 'obj' markers in the grid accordingly.
        """
        # Validate bounds for each entity
        for name, x, y in (
            ("start", start_x, start_y),
            ("object", object_x, object_y),
            ("target", target_x, target_y),
        ):
            if not (0 <= x < self.cols and 0 <= y < self.rows):
                raise ValueError(f"{name} out of bounds")

        # Assign positions (stored as (x, y))
        self.start = (start_x, start_y)
        self.object = (object_x, object_y)
        self.target = (target_x, target_y)

        # Write obj flags into grid
        self._apply_entity_flags()

    # -------------------- JSON I/O --------------------

    def SaveMaze(self, filename: str) -> None:
        """
        Save the current maze to JSON.
        NOTE: This version writes the maze as a FLAT row-major list (not 2-D),
        exactly matching your current implementation used by the rest of the project.
        """
        data = {
            "rows": self.rows,
            "cols": self.cols,
            "start": list(self.start) if self.start else None,   # stored as [x, y]
            "object": list(self.object) if self.object else None,
            "target": list(self.target) if self.target else None,
            "maze": [self.maze[r][c] for r in range(self.rows) for c in range(self.cols)],
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def LoadMaze(
        self,
        source: Optional[str] = None,
        seed: Optional[int] = None,
        braid_factor: float = 0.15,
    ) -> None:
        """
        Load from a JSON file (if 'source' is a path string), or generate a random maze.
        When loading JSON:
          - rows/cols are overridden
          - start/object/target are overridden
          - entire maze data is overridden
        Supports both a FLAT list of cells and a 2-D list of rows.
        `braid_factor` is only used when generating (it is ignored when a
        JSON file is loaded, since the file already fixes the layout).
        """
        if isinstance(source, str):
            # Load JSON and rebuild grid
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.rows = int(data["rows"])
            self.cols = int(data["cols"])

            # Rebuild grid from either a flat list or a 2-D array
            if isinstance(data.get("maze"), list) and data["maze"] and isinstance(data["maze"][0], dict):
                # Flat list → reshape row-major
                it = iter(data["maze"])
                self.maze = [[next(it) for _ in range(self.cols)] for _ in range(self.rows)]
            else:
                # 2-D form already
                raw = data["maze"]
                self.maze = [[raw[r][c] for c in range(self.cols)] for r in range(self.rows)]

            # Positions are stored as [x, y]
            self.start = tuple(data.get("start")) if data.get("start") else None
            self.object = tuple(data.get("object")) if data.get("object") else None
            self.target = tuple(data.get("target")) if data.get("target") else None

            # Ensure 'coords' field is consistent (useful for debugging)
            for r in range(self.rows):
                for c in range(self.cols):
                    self.maze[r][c]["coords"] = [r, c]
        else:
            # Build a fresh random maze if no file was provided
            self.generate_maze(seed=seed, braid_factor=braid_factor)

    # -------------------- Internal helpers --------------------

    def _delta(self, direction: int) -> Tuple[int, int]:
        """
        Convert a direction constant into a (dr, dc) step in (row, col) space:
          RIGHT → (0, +1), TOP → (-1, 0), LEFT → (0, -1), DOWN → (+1, 0)
        """
        if direction == RIGHT:
            return 0, 1
        if direction == TOP:
            return -1, 0
        if direction == LEFT:
            return 0, -1
        if direction == DOWN:
            return 1, 0
        raise ValueError("invalid direction")

    def _place_entities_randomly(self, rng: random.Random) -> None:
        """
        Ensure start/object/target all exist and are mutually distinct.
        If any are missing (None), assign a random free (x, y) position.
        """
        def rand_pos() -> Tuple[int, int]:
            # Return a random (x, y) where x = col, y = row
            return rng.randrange(self.cols), rng.randrange(self.rows)

        used = set()

        def pick(distinct_from: set) -> Tuple[int, int]:
            """Pick a random (x, y) not in 'distinct_from', insert it, and return it."""
            while True:
                x, y = rand_pos()
                if (x, y) not in distinct_from:
                    distinct_from.add((x, y))
                    return (x, y)

        # Start
        if self.start is None:
            self.start = pick(used)
        else:
            used.add(self.start)

        # Object
        if self.object is None:
            self.object = pick(used)
        else:
            used.add(self.object)

        # Target
        if self.target is None:
            self.target = pick(used)
        else:
            used.add(self.target)

    def _apply_entity_flags(self) -> None:
        """
        Write the object/target markers into the grid:
          - obj =  1 at the object cell
          - obj = -1 at the target cell
          - obj =  0 elsewhere
        """
        # Clear all markers
        for r in range(self.rows):
            for c in range(self.cols):
                self.maze[r][c]["obj"] = 0

        # Mark object if within bounds
        if self.object:
            ox, oy = self.object  # (x, y) = (col, row)
            if 0 <= oy < self.rows and 0 <= ox < self.cols:
                self.maze[oy][ox]["obj"] = 1

        # Mark target if within bounds
        if self.target:
            tx, ty = self.target  # (x, y) = (col, row)
            if 0 <= ty < self.rows and 0 <= tx < self.cols:
                self.maze[ty][tx]["obj"] = -1
