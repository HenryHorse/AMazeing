import numpy as np
from collections import deque
from models.generator import generate_prim_algo
from models.random_mutator import is_solvable
from models.CNN_mutator import CNNUniqueMutatorAgent, shortest_path_length
from visualize import astar_path

MOVES = [(-1,0),(1,0),(0,-1),(0,1)]

def get_shortest_path(maze, start, goal):
    """
    Returns the sequence of positions along one shortest path via BFS.
    """
    rows, cols = maze.shape
    queue = deque([start])
    parent = {start: None}
    while queue:
        curr = queue.popleft()
        if curr == goal:
            path = []
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            return list(reversed(path))
        r, c = curr
        for dr, dc in MOVES:
            nxt = (r + dr, c + dc)
            if (0 <= nxt[0] < rows and 0 <= nxt[1] < cols and
                maze[nxt] == 0 and nxt not in parent):
                parent[nxt] = curr
                queue.append(nxt)
    return None

def simulate_one_episode(mutator, rows=23, cols=23):
    """
    Generates a solvable maze, computes baseline A* length, then
    simulates solver vs. mutator (mutates every 3 solver steps)
    and returns (baseline_len, actual_len).
    """
    # reset unique-swap history
    mutator.reset()

    # 1) generate a random solvable maze
    while True:
        maze, start, goal = generate_prim_algo(rows, cols)
        if is_solvable(maze, start=start, goal=goal):
            break

    # 2) baseline shortest-path length with no mutations
    baseline = shortest_path_length(maze, start, goal)
    assert baseline is not None

    # 3) simulate solver+mutator
    maze_sim = maze.copy()
    pos = start
    steps = 0

    while pos != goal:
        # every 3 moves, let mutator apply one swap
        if steps > 0 and steps % 3 == 0:
            # PPO mutator returns (new_maze, logp, value, ent)
            maze_sim, _, _, _ = mutator.mutate(maze_sim, pos, goal)

        # next solver move: take one step along current shortest path
        path = get_shortest_path(maze_sim, pos, goal)
        if path is None or len(path) < 2:
            # unsolvable or stuck — abort
            break

        pos = path[1]
        steps += 1

        # safety cap to avoid infinite loops
        if steps > rows * cols * 10:
            break

    actual = steps
    return baseline, actual

def evaluate_mutator(mutator, episodes=1000, rows=23, cols=23):
    deltas = []
    for i in range(episodes):
        base, actual = simulate_one_episode(mutator, rows, cols)
        deltas.append(actual - base)
        print(f"Episode {i+1}/{episodes} — avg Δ so far: {np.mean(deltas):.2f}, last Δ: {deltas[-1]}")
    return np.mean(deltas), np.std(deltas)

if __name__ == "__main__":
    # instantiate your trained PPO mutator here
    mutator = CNNUniqueMutatorAgent(rows=23, cols=23)
    try:
        mutator.load()
        print("Loaded trained PPO mutator.")
    except FileNotFoundError:
        print("No trained mutator found; please train first.")

    mean_delta, std_delta = evaluate_mutator(mutator, episodes=1000, rows=23, cols=23)
    print(f"\nFinal over 100 episodes — avg delay: {mean_delta:.2f} moves (±{std_delta:.2f})")
