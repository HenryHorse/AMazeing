from stable_baselines3 import PPO
from models.solver import GNNSolverAgent
from models.mutator import MazeMutatorAgent
from models.generator import generate_maze_dfs_backtracker
from models.random_mutator import is_solvable

solver = GNNSolverAgent()
mutator = MazeMutatorAgent()  # now uses the new GNNMutatorPolicy.forward()

for epoch in range(1000):
    maze, start, goal = generate_maze_dfs_backtracker(10, 10)
    maze[start] = maze[goal] = 0

    # 1) Get the solver's baseline:
    _, _, base_path = solver.run_episode(maze, start, goal)
    base_len = len(base_path) if (base_path and base_path[-1] == goal) else 1000

    # 2) Mutator tries two swaps
    log_probs_mutator = []

    action, log_prob = mutator.select_action(maze, start, start, goal)
    new_maze = mutator.mutate(maze, action)
    if is_solvable(new_maze, start, goal):
        maze = new_maze
        log_probs_mutator.append(log_prob)
    else:
        # Penalize the mutator if it made the maze unsolvable
        mutator.update([log_prob], [-10.0])

    # 3) Solver then tries the (possibly mutated) maze
    log_probs_solver, rewards_solver, path = solver.run_episode(maze, start, goal)
    solver.update(log_probs_solver, rewards_solver)

    # 4) Compute mutator's “difficulty” reward
    if (path and path[-1] == goal):
        mutator_reward = max(0, len(path) - base_len)
    else:
        mutator_reward = 5.0

    mutator.update(log_probs_mutator, [mutator_reward] * len(log_probs_mutator))

    print(f"[{epoch:3d}] Solver {'solved' if (path and path[-1]==goal) else 'failed'} "
          f"in {len(path)} steps. Mutator reward: {mutator_reward}")