# file: maze_viewer.py
# Tkinter Maze Viewer with animation support.
# Shows walls (order: [Right, Top, Left, Down]), visited cells, final path, and S/O/T dots.
# Note: MapServer stores positions as (x=col, y=row); drawing converts to (row=y, col=x).

import tkinter as tk
from typing import List, Optional, Tuple
from mapserver import MapServer, RIGHT, TOP, LEFT, DOWN


class MazeViewer:
    def __init__(
        self,
        mapserver: MapServer,
        cell_size: int = 40,
        wall_width: int = 2,
        padding: int = 10,
        title_suffix: str = "",
    ) -> None:
        """Keep constructor behavior unchanged; compute fixed canvas size from map size."""
        self.m = mapserver
        self.cell_size = int(cell_size)      # pixel size of one cell
        self.wall_width = int(wall_width)    # line width for walls
        self.padding = int(padding)          # border around the grid
        self.title_suffix = title_suffix     # optional text in window title

        rows, cols = self.m.getSize()
        self.width = self.padding * 2 + cols * self.cell_size
        self.height = self.padding * 2 + rows * self.cell_size

        # Lazily created Tk window/canvas; set by open()/show()
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._banner_text: str = ""          # single-line banner drawn on top-left

    # ---------------- geometry ----------------

    def _cell_bounds(self, r: int, c: int) -> Tuple[int, int, int, int]:
        """Pixel rectangle for cell (row=r, col=c). Top-left (x1,y1) to bottom-right (x2,y2)."""
        x1 = self.padding + c * self.cell_size
        y1 = self.padding + r * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        return x1, y1, x2, y2

    def _cell_center(self, r: int, c: int) -> Tuple[int, int]:
        """Pixel center of cell (r,c) — used to draw polyline for the final path."""
        x1, y1, x2, y2 = self._cell_bounds(r, c)
        return (x1 + x2) // 2, (y1 + y2) // 2

    # ---------------- entities ----------------

    def _draw_entities(self, canvas: tk.Canvas) -> None:
        """Draw Start (green), Object (blue), Target (red) as filled circles."""
        def dot(x: int, y: int, color: str) -> None:
            # MapServer gives (x=col, y=row). Convert to drawing (row=y, col=x).
            r = int(self.cell_size * 0.3)            # radius in pixels
            x1, y1, x2, y2 = self._cell_bounds(y, x) # convert to cell rectangle
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # cell center
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="")

        if self.m.start is not None:
            sx, sy = self.m.start
            dot(sx, sy, "green")  # Start
        if self.m.object is not None:
            ox, oy = self.m.object
            dot(ox, oy, "blue")   # Object
        if self.m.target is not None:
            tx, ty = self.m.target
            dot(tx, ty, "red")    # Target

    # ---------------- core drawing ----------------

    def draw_maze(self, visited: List[Tuple[int, int]], path: List[Tuple[int, int]]) -> None:
        """
        Draw walls for every cell; then overlay visited cells and final path.
        - visited: list of (row, col) expanded nodes (light yellow fill)
        - path:    ordered list of (row, col) solution nodes (green fill + polyline)
        """
        rows, cols = self.m.getSize()

        # Walls: respect order [Right, Top, Left, Down]
        for r in range(rows):
            for c in range(cols):
                walls = self.m.getRoom(r, c)["walls"]
                x1, y1, x2, y2 = self._cell_bounds(r, c)
                if walls[RIGHT] == 1:
                    self._canvas.create_line(x2, y1, x2, y2, width=self.wall_width)
                if walls[TOP] == 1:
                    self._canvas.create_line(x1, y1, x2, y1, width=self.wall_width)
                if walls[LEFT] == 1:
                    self._canvas.create_line(x1, y1, x1, y2, width=self.wall_width)
                if walls[DOWN] == 1:
                    self._canvas.create_line(x1, y2, x2, y2, width=self.wall_width)

        # Visited nodes (navigation history)
        if visited:
            inset = max(1, self.cell_size // 10)  # small inset to keep wall lines visible
            for (r, c) in visited:
                x1, y1, x2, y2 = self._cell_bounds(r, c)
                self._canvas.create_rectangle(
                    x1 + inset, y1 + inset, x2 - inset, y2 - inset,
                    fill="#fff59d", outline=""
                )

        # Final path overlay (filled cells + line through centers)
        if path:
            inset = max(1, self.cell_size // 6)
            for (r, c) in path:
                x1, y1, x2, y2 = self._cell_bounds(r, c)
                self._canvas.create_rectangle(
                    x1 + inset, y1 + inset, x2 - inset, y2 - inset,
                    fill="#81c784", outline=""
                )
            pts: List[int] = []
            for (r, c) in path:
                cx, cy = self._cell_center(r, c)
                pts.extend([cx, cy])
            if len(pts) >= 4:  # need at least two points to draw a line
                self._canvas.create_line(*pts, width=max(2, self.wall_width), fill="#2e7d32")

        # Draw S/O/T dots on top of overlays
        self._draw_entities(self._canvas)

        # Single-line banner (status/title)
        if self._banner_text:
            self._canvas.create_text(
                self.padding + 6,
                self.padding + 10,
                text=self._banner_text,
                anchor="w",
                font=("Arial", max(10, self.cell_size // 3), "bold"),
                fill="#1b5e20",
            )

    # ---------------- snapshot (non-animated) ----------------

    def show(self, visited: List[Tuple[int, int]], path: List[Tuple[int, int]], title: str = "") -> None:
        """Create window, draw once (no animation), and enter Tk mainloop."""
        rows, cols = self.m.getSize()
        self._root = tk.Tk()
        suffix = f" – {rows}x{cols} {title or self.title_suffix}"
        self._root.title("Maze Viewer (Comparison)" + suffix)
        self._canvas = tk.Canvas(self._root, width=self.width, height=self.height, bg="white")
        self._canvas.pack()
        self.draw_maze(visited=visited, path=path)
        self._root.mainloop()

    # ---------------- live animation support ----------------

    def open(self, window_title: Optional[str] = None) -> None:
        """
        Prepare window/canvas for step-by-step animation.
        Caller controls frames via redraw_step; we do NOT call mainloop here.
        """
        if self._root is not None:
            return  # already opened
        rows, cols = self.m.getSize()
        self._root = tk.Tk()
        suffix = f" – {rows}x{cols} {self.title_suffix}"
        self._root.title(window_title or ("Maze Viewer (Animated)" + suffix))
        self._canvas = tk.Canvas(self._root, width=self.width, height=self.height, bg="white")
        self._canvas.pack()
        self.redraw_step([], [], delay_ms=0)  # initial clear frame

    def redraw_step(self, visited: List[Tuple[int, int]], path: List[Tuple[int, int]], delay_ms: int) -> None:
        """
        One animation frame:
          1) clear canvas,
          2) draw maze + overlays,
          3) process UI events,
          4) wait 'delay_ms' milliseconds.
        """
        if self._canvas is None:
            return
        self._canvas.delete("all")                 # clear previous frame
        self.draw_maze(visited=visited, path=path) # draw current frame
        self._canvas.update()                      # process UI events / refresh
        if delay_ms and delay_ms > 0:
            self._canvas.after(int(delay_ms))      # paced animation

    def set_banner(self, text: str) -> None:
        """Set single-line banner (drawn in draw_maze)."""
        self._banner_text = text

    def mainloop(self) -> None:
        """Enter Tk main event loop (blocks until window closes)."""
        if self._root is not None:
            self._root.mainloop()
