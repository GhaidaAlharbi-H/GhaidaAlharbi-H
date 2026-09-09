# AI381 — Robot Maze Navigation: BFS vs DFS vs Greedy vs A*

A grid-maze delivery task used to compare four classic search strategies under
identical conditions.

## Problem statement

A robot is placed in a rectangular maze of walled cells. It must complete a
three-phase delivery run:

1. **Find the object** — travel from `Start` to the cell holding the object.
2. **Deliver it** — carry the object from there to the `Target` cell.
3. **Go home** — return from `Target` to `Start`.

The robot has no map of its own: it perceives the maze only through
`MapServer.getRoom(row, col)`, which reveals the four walls and the movement
cost of a single cell, and `MapServer.getSize()`. Movement is four-directional
and every cell costs 1, so the shortest path is simply the fewest moves.

The question the project answers: **given the same maze, the same start, object
and target, how do BFS, DFS, Greedy Best-First and A\* differ in path quality
and in search effort?**

### Why the maze is *braided*

A textbook "recursive backtracker" generates a **perfect** maze — a spanning
tree with exactly one path between any two cells. On such a maze all four
algorithms are forced down the same corridor and return an *identical* path, so
the comparison measures nothing but timing noise.

`generate_maze()` therefore takes a `braid_factor` (default `0.15`): after
carving, that fraction of the walls still standing is removed at random, opening
both reciprocal walls each time. The loops this creates give the algorithms real
choices, and the differences below appear. Set `--braid-factor 0.0` to reproduce
the degenerate perfect-maze behaviour.

## Algorithm comparison

Averages over 20 seeds on a 12x16 braided maze (`braid_factor = 0.15`), summed
across all three phases. *Path* = moves taken; *Expanded* = unique cells the
search dequeued.

| Algorithm          | Strategy                        | Optimal? | Mean path | Mean expanded | Notes |
|--------------------|---------------------------------|----------|-----------|---------------|-------|
| **BFS**            | Frontier by depth (FIFO queue)  | Yes      | **43.3**  | 153.7         | Optimal, but explores nearly the whole maze |
| **DFS**            | Frontier by depth (LIFO stack)  | No       | 133.7     | 157.8         | ~3x longer paths for no saving in effort |
| **Greedy Best-First** | Lowest Manhattan `h(n)`      | No       | 52.9      | **58.1**      | Cheapest search, paths ~22% above optimal |
| **A\***            | Lowest `f(n) = g(n) + h(n)`     | Yes      | **43.3**  | 86.8          | Optimal like BFS at ~44% less search effort |

**Takeaway:** BFS and A\* always tie on path length (both are optimal on a
uniform-cost grid), but A\* reaches the goal after expanding far fewer cells.
DFS finds a path quickly yet wanders badly. Greedy is the cheapest to run but
gives up optimality. A\* is the best overall trade-off, which is what
`select_best()` reports for the default seed.

A single run of the default configuration looks like this:

```
=== Algorithm Comparison on SAME Maze ===
Algo      Expanded  PathLen   Cost  Find(s)  ToTgt(s) ToStart(s)  Total(s)
--------------------------------------------------------------------------
BFS            173       34     34   0.0002    0.0002     0.0001    0.0005
DFS            147      178    178   0.0003    0.0001     0.0002    0.0006
GREEDY          44       42     42   0.0001    0.0001     0.0001    0.0002
ASTAR           65       34     34   0.0001    0.0003     0.0001    0.0005

Selected BEST algorithm: ASTAR (criterion: cost, tie-breakers: ('path_len', 'expanded', 't_total'))
```

## How to run

Python 3.8+ is required; the project itself needs nothing outside the standard
library.

```bash
# Full comparison, then animate the winning algorithm (needs tkinter)
python compare_algorithms.py

# Comparison only — no GUI, safe on a server or over SSH
python compare_algorithms.py --headless

# Change the maze
python compare_algorithms.py --headless --seed 42 --rows 20 --cols 30

# Reproduce the perfect-maze bug: all four algorithms return the same path
python compare_algorithms.py --headless --braid-factor 0.0
```

All options (the module constants at the top of `compare_algorithms.py` are the
defaults):

| Flag | Default | Meaning |
|------|---------|---------|
| `--rows N` | 12 | Maze rows |
| `--cols N` | 16 | Maze columns |
| `--seed N` | 121 | RNG seed freezing the layout |
| `--braid-factor F` | 0.15 | Fraction of remaining walls removed to create loops |
| `--json-file PATH` | `maze_compare.json` | Where the shared maze is persisted |
| `--headless` | off | Skip the animation |

The script writes the generated maze to `maze_compare.json` (so every algorithm
provably runs on the same layout) and appends one timestamped line per algorithm
to `run_log.txt`.

If tkinter is missing or no display is available, the comparison still completes
and only the animation is skipped, with a warning explaining why.

### Tests

```bash
pip install -r requirements.txt
pytest -q
```

The suite covers optimality (BFS == A\*, DFS >= BFS), path validity (every path
is contiguous and never crosses a wall), maze generation (perfect vs braided,
full connectivity, large grids without recursion limits) and JSON save/load
round-tripping.

## Animation

Running without `--headless` opens a tkinter window that animates the winning
algorithm through all three phases: expanded cells fill in pale yellow as the
search runs, the final route is drawn in green, and Start / Object / Target are
the green, blue and red dots.

<!-- TODO: replace with a screenshot or GIF of the animation, e.g. docs/animation.gif -->
![Animation of the best algorithm solving the maze](docs/animation.gif)

## Project layout

| File | Purpose |
|------|---------|
| `mapserver.py` | Maze storage, generation (carve + braid), JSON save/load |
| `robot_agent.py` | The robot and the four planners (BFS, DFS, Greedy, A\*) |
| `maze_viewer.py` | tkinter viewer with step-by-step animation |
| `compare_algorithms.py` | Runs the comparison, logs metrics, animates the winner |
| `tests/test_algorithms.py` | pytest suite |

### Conventions

Two conventions are used consistently throughout and should not be changed:

- **Coordinates.** Public positions (`start`, `object`, `target`) are `(x, y) =
  (col, row)`. Internal grid access is `maze[row][col]`, and planners work in
  `(row, col)`.
- **Directions.** Wall lists are always ordered `[RIGHT, TOP, LEFT, DOWN]`
  (`0, 1, 2, 3`), with `1` = wall present and `0` = open passage. Walls are
  reciprocal: opening one always opens its twin in the neighbouring cell.
