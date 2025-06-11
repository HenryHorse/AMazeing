# models/heuristic_mutator.py
import random
from collections import deque
import numpy as np

from models.random_mutator import is_solvable
from models.helpers import maze_to_tensor

# reuse BFS to get actual path, not just length
def get_shortest_path(maze, start, goal):
    rows, cols = maze.shape
    visited = set([start])
    queue = deque([start])
    parent = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            # reconstruct path
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))
        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr,nc] == 0:
                nxt = (nr, nc)
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = cur
                    queue.append(nxt)
    return None


def path_length(maze, start, goal):
    """
    Compute the length (in number of moves) of the shortest path from start to goal
    in a binary grid `maze` (0=free, 1=wall). Returns an int or None if unsolvable.
    """
    rows, cols = maze.shape
    queue = deque([(start, 0)])     # (position, distance_so_far)
    visited = set([start])
    moves = [(-1,0), (1,0), (0,-1), (0,1)]

    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows and 0 <= nc < cols and
                maze[nr][nc] == 0 and
                (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))

    # no path found
    return None




class TrueRandomMutator:
    """
    A truly random mutator: uniformly samples (remove, add) pairs
    across the whole maze, with a solvability check.
    """
    def __init__(self, max_attempts: int = 1000):
        # how many random trials before giving up and returning the original maze
        self.max_attempts = max_attempts

    def mutate(self, maze: np.ndarray, solver_pos: tuple, goal: tuple) -> np.ndarray:
        """
        Returns a new maze where:
          - one random wall cell has been removed, and
          - one random empty cell has been walled,
        chosen uniformly at random (with retry up to max_attempts),
        and only accepted if the result remains solvable.
        """
        rows, cols = maze.shape
        walls   = list(zip(* (maze == 1).nonzero()))
        empties = list(zip(* (maze == 0).nonzero()))

        for _ in range(self.max_attempts):
            # pick a random wall to remove
            rem = random.choice(walls)
            # pick a random empty cell to add
            add = random.choice(empties)

            # never block the solver’s current cell or the goal
            if add == solver_pos or add == goal:
                continue

            # apply the swap
            trial = maze.copy()
            trial[rem]  = 0
            trial[add]  = 1

            # accept only if still solvable
            if is_solvable(trial, solver_pos, goal):
                return trial

        # if we never found a valid swap, return the original maze
        return maze

class SimpleHeuristicMutator:
    """
    A simple heuristic mutator that:
      1) Adds a wall on the solver’s current shortest path
      2) Removes a wall elsewhere to keep solvability
      But never repeats the exact same (remove,add) swap twice.
    """
    def __init__(self):
        # holds ( (wr,wc), (ar,ac) ) pairs we’ve already used
        self.seen_swaps = set()

    def reset(self):
        """Clear history so you can replay fresh episodes."""
        self.seen_swaps.clear()

    def mutate(self, maze: np.ndarray, solver_pos: tuple, goal: tuple) -> np.ndarray:
        # 1) find the solver’s current shortest path
        path = get_shortest_path(maze, solver_pos, goal)
        if not path or len(path) < 3:
            return maze  # nothing meaningful to block

        # shuffle the interior path cells so we try them in random order
        blocks = path[1:-1]
        random.shuffle(blocks)

        # for each candidate block cell, try to find a removal that is both:
        #   a) not already in seen_swaps
        #   b) keeps the maze solvable
        for block in blocks:
            # tentative add
            trial_add = maze.copy()
            trial_add[block] = 1

            # list existing walls in that trial
            walls = list(zip(* (trial_add == 1).nonzero()))
            random.shuffle(walls)

            for wr, wc in walls:
                if (wr, wc) == block:
                    continue

                swap = ((wr, wc), block)
                if swap in self.seen_swaps:
                    continue  # skip repeats

                # try removing this wall
                trial = trial_add.copy()
                trial[wr, wc] = 0

                if not is_solvable(trial, solver_pos, goal):
                    continue  # can’t break solvability

                # SUCCESS: record & return
                self.seen_swaps.add(swap)
                return trial

        # no fresh, solvable swap found
        return maze
    
