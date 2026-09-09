# file: compare_algorithms.py
# Multi-Algorithm Comparison + Auto-Run Best with LIVE ANIMATION (single-line title).
# Compares BFS, DFS, GREEDY, ASTAR on the same maze (no user input).
#
# What this script does:
# 1) Generates one BRAIDED maze (fixed seed) and saves it to JSON. Braiding adds
#    loops so the four algorithms actually have different paths to choose from.
# 2) Loads that SAME maze for each algorithm (BFS/DFS/Greedy/A*) and runs:
#       - find object
#       - move to target
#       - return to start
#    while collecting metrics (expanded nodes, path length, cost, timings).
# 3) Prints a comparison table to the terminal and appends detailed lines to run_log.txt.
# 4) Selects the "best" algorithm by (SELECT_PRIMARY, then TIE_BREAKERS).
# 5) Animates the FULL run of the best algorithm (live step-by-step visualization),
#    unless --headless is given or tkinter is unavailable.
#
# Usage:
#   python compare_algorithms.py [--headless] [--seed N] [--rows N] [--cols N]
#                                [--braid-factor F] [--json-file PATH]

import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple

from mapserver import MapServer
from robot_agent import Robot

# NOTE: maze_viewer imports tkinter, which is missing on headless machines.
# It is imported lazily inside main() so the comparison always runs.

# ---------------- Configuration ----------------
ROWS = 12                # maze rows (fixed; no user input)
COLS = 16                # maze cols (fixed; no user input)
SEED = 121               # RNG seed to freeze the maze layout
                         # (seed 123 happens to produce a layout where all four
                         #  algorithms coincide; 121 shows them diverging)
JSON_FILE = "maze_compare.json"  # path to persist the generated maze

# Fraction of the walls left standing after carving that are removed at random.
# 0.0 => perfect (loop-free) maze: exactly one path exists between any two cells,
#        so BFS/DFS/Greedy/A* all return the SAME path and the comparison is
#        meaningless. Anything above 0.0 adds loops and real choices.
BRAID_FACTOR = 0.15

ALGORITHMS = ("bfs", "dfs", "greedy", "astar")  # algorithms to compare

# Best-selection rule (lower is better). Primary then tie-breakers in order.
# 'expanded' comes before 't_total' so two algorithms that find equally short
# paths (e.g. BFS and A*) are separated by search effort rather than by
# millisecond timing noise, which is not reproducible between runs.
SELECT_PRIMARY = "cost"
TIE_BREAKERS = ("path_len", "expanded", "t_total")

# Animation settings for the final, best algorithm
RUN_BEST_WITH_ANIMATION = True
ANIM_DELAY_MS = 30
ANIM_CELL_SIZE = 36
ANIM_WALL_WIDTH = 2
ANIM_PADDING = 12


# ---------------- Helpers ----------------
class DummyViewer:
    """No-op viewer to keep planner API the same for headless timing."""
    def redraw_step(self, visited, path, delay_ms: int) -> None:
        return


def total_cost(m: MapServer, path: List[Tuple[int, int]]) -> int:
    """Sum of cell costs along the path (excluding the starting node)."""
    return sum(m.getRoom(r, c)["cost"] for (r, c) in path[1:]) if path else 0


def run_full_task(m: MapServer, method: str) -> Dict:
    """
    Run one algorithm end-to-end on the given maze:
      1) find object
      2) move to target
      3) return to start
    Uses DummyViewer to avoid animation overhead during comparison.
    Returns a dict with metrics for later printing/logging/selection.
    """
    viewer = DummyViewer()
    robot = Robot(m)

    # Timed phases
    t0 = time.perf_counter()
    robot.find_object_animated(viewer, delay_ms=0, method=method)
    t1 = time.perf_counter()

    robot.move_to_target_animated(viewer, delay_ms=0, method=method)
    t2 = time.perf_counter()

    robot.return_to_start_animated(viewer, delay_ms=0, method=method)
    t3 = time.perf_counter()

    # Collect final traces and metrics
    path = list(robot.path_history)
    visited = list(robot.visited_nodes)
    rows, cols = m.getSize()

    return {
        "method": method.upper(),
        "size": (rows, cols),
        "visited": visited,
        "path": path,
        "path_len": max(0, len(path) - 1),  # number of moves
        "expanded": len(set(visited)),      # unique expanded nodes
        "cost": total_cost(m, path),
        "t_find": t1 - t0,
        "t_target": t2 - t1,
        "t_return": t3 - t2,
        "t_total": t3 - t0,
        "final_pos": path[-1] if path else None,
    }


def print_table(results: List[Dict]) -> None:
    """Pretty terminal table summarizing core metrics for each algorithm."""
    print("\n=== Algorithm Comparison on SAME Maze ===")
    header = f"{'Algo':<8} {'Expanded':>9} {'PathLen':>8} {'Cost':>6} {'Find(s)':>8} {'ToTgt(s)':>9} {'ToStart(s)':>10} {'Total(s)':>9}"
    print(header)
    print("-" * len(header))
    for d in results:
        print(f"{d['method']:<8} {d['expanded']:>9} {d['path_len']:>8} {d['cost']:>6} "
              f"{d['t_find']:>8.4f} {d['t_target']:>9.4f} {d['t_return']:>10.4f} {d['t_total']:>9.4f}")
    print()


def log_results(results: List[Dict], seed: int) -> None:
    """Append a concise, timestamped line per algorithm to run_log.txt."""
    with open("run_log.txt", "a", encoding="utf-8") as f:
        for d in results:
            rows, cols = d["size"]
            f.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"algo={d['method']} size={rows}x{cols} "
                f"expanded={d['expanded']} path_len={d['path_len']} cost={d['cost']} "
                f"t_find={d['t_find']:.6f}s t_target={d['t_target']:.6f}s t_return={d['t_return']:.6f}s "
                f"t_total={d['t_total']:.6f}s seed={seed} final_pos={d['final_pos']}\n"
            )


def select_best(results: List[Dict]) -> Dict:
    """
    Select the best result by the configured keys.
    Order: SELECT_PRIMARY, then TIE_BREAKERS. 'min' on the tuple comparison.
    """
    keys = (SELECT_PRIMARY,) + tuple(TIE_BREAKERS)
    def key_fn(d: Dict):
        return tuple(d[k] for k in keys)
    return min(results, key=key_fn)


# ---------------- CLI ----------------
def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line options. The module constants are used as defaults."""
    parser = argparse.ArgumentParser(
        description="Compare BFS, DFS, Greedy Best-First and A* on one maze.",
    )
    parser.add_argument("--rows", type=int, default=ROWS,
                        help=f"maze rows (default: {ROWS})")
    parser.add_argument("--cols", type=int, default=COLS,
                        help=f"maze cols (default: {COLS})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"RNG seed freezing the maze layout (default: {SEED})")
    parser.add_argument("--braid-factor", type=float, default=BRAID_FACTOR, metavar="F",
                        help="fraction of remaining walls removed to create loops; "
                             f"0.0 gives a perfect maze (default: {BRAID_FACTOR})")
    parser.add_argument("--json-file", default=JSON_FILE, metavar="PATH",
                        help=f"where to persist the generated maze (default: {JSON_FILE})")
    parser.add_argument("--headless", action="store_true",
                        help="skip the tkinter animation of the best algorithm "
                             "(use on machines with no display)")
    return parser.parse_args(argv)


# ---------------- Main ----------------
def main(argv=None) -> None:
    args = parse_args(argv)

    # 1) Freeze maze once (deterministic with the seed), persist to JSON
    base = MapServer(rows=args.rows, cols=args.cols)
    base.LoadMaze(seed=args.seed, braid_factor=args.braid_factor)  # generate
    base.SaveMaze(args.json_file)    # save so every algorithm loads the identical maze
    print(f"[INFO] Maze {args.rows}x{args.cols} seed={args.seed} "
          f"braid_factor={args.braid_factor} -> {args.json_file}")

    # 2) Compare algorithms headlessly on the SAME maze
    results: List[Dict] = []
    for alg in ALGORITHMS:
        m = MapServer(rows=1, cols=1)  # placeholder; JSON will override rows/cols/data
        m.LoadMaze(args.json_file)     # load same layout each time
        results.append(run_full_task(m, method=alg))

    print_table(results)                    # terminal summary
    log_results(results, seed=args.seed)    # append to run_log.txt

    # 3) Pick best by configured ranking rule
    best = select_best(results)
    print(f"Selected BEST algorithm: {best['method']} (criterion: {SELECT_PRIMARY}, tie-breakers: {TIE_BREAKERS})")

    # 4) Animate the best run end-to-end (live visualization)
    if args.headless or not RUN_BEST_WITH_ANIMATION:
        reason = "--headless requested" if args.headless else "RUN_BEST_WITH_ANIMATION is False"
        print(f"[INFO] Comparison complete; animation skipped ({reason}).")
        return

    # tkinter may be absent (headless server, minimal Python build). The comparison
    # above has already finished, so a missing GUI must not fail the whole run.
    try:
        from maze_viewer import MazeViewer, tk
    except ImportError as exc:
        print("[WARN] Comparison completed successfully, but the animation was "
              f"skipped: could not import the tkinter viewer ({exc}).")
        print("[WARN] Install tkinter (e.g. 'apt install python3-tk') or re-run "
              "with --headless to silence this message.")
        return

    # Reload the same maze to animate from a clean state
    m = MapServer(rows=1, cols=1)
    m.LoadMaze(args.json_file)

    # Create viewer window/canvas once, then feed frames from robot planners.
    # tkinter can be installed yet still have no display to draw on (e.g. SSH
    # without X forwarding); that is another "skip the animation" case.
    viewer = MazeViewer(
        m,
        cell_size=ANIM_CELL_SIZE,
        wall_width=ANIM_WALL_WIDTH,
        padding=ANIM_PADDING,
        title_suffix=f"{best['method']} (BEST) seed={args.seed}",
    )
    try:
        viewer.open(window_title=f"Maze Viewer – Animated ({best['method']})")
    except tk.TclError as exc:
        print("[WARN] Comparison completed successfully, but the animation was "
              f"skipped: tkinter could not open a window ({exc}).")
        print("[WARN] Re-run with --headless to silence this message.")
        return

    robot = Robot(m)

    # Phase 1: start → object
    viewer.set_banner(f"Algorithm: {best['method']} — Searching for object")
    robot.find_object_animated(viewer, delay_ms=ANIM_DELAY_MS, method=best["method"].lower())

    # Phase 2: object → target
    viewer.set_banner(f"Algorithm: {best['method']} — Delivering to target")
    robot.move_to_target_animated(viewer, delay_ms=ANIM_DELAY_MS, method=best["method"].lower())

    # Phase 3: target → start
    viewer.set_banner(f"Algorithm: {best['method']} — Returning to start")
    robot.return_to_start_animated(viewer, delay_ms=ANIM_DELAY_MS, method=best["method"].lower())

    # Final frame with full overlays
    viewer.set_banner(f"Algorithm: {best['method']} — COMPLETE")
    viewer.redraw_step(visited=robot.visited_nodes, path=robot.path_history, delay_ms=0)
    print(f"[INFO] Animated run displayed using {best['method']} – close the window to exit.")
    viewer.mainloop()


if __name__ == "__main__":
    main()
