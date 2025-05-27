import gym
import numpy as np
from gym import spaces
from models.generator import generate_random_maze
from models.random_mutator import is_solvable
import networkx as nx


def solver_path_length(maze, start=(0, 0), goal=None):
        if goal is None:
            goal = (maze.shape[0] - 1, maze.shape[1] - 1)

        G = nx.Graph()
        rows, cols = maze.shape
        for r in range(rows):
            for c in range(cols):
                if maze[r][c] == 0:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                            G.add_edge((r, c), (nr, nc))

        try:
            path = nx.shortest_path(G, start, goal)
            return len(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None  # unsolvable



class MazeMutationEnv(gym.Env):
    """
    Gym environment for training a maze-mutating agent.
    The agent selects a wall to move to an empty space.
    Reward is +1 for a valid move (still solvable), -10 if it breaks solvability.
    """
    def __init__(self, rows=10, cols=10):
        super(MazeMutationEnv, self).__init__()
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)

        # Observation is the flattened maze (0s and 1s)
        self.observation_space = spaces.Box(low=0, high=1, shape=(rows * cols,), dtype=np.uint8)

        # Action: (from_index, to_index) as a flat pair
        self.action_space = spaces.MultiDiscrete([rows * cols, rows * cols])

        self.maze = None
        self.reset()
    

    def reset(self):
        self.maze = self._generate_solvable_maze()
        return self.maze.flatten()

    def step(self, action):
        from_idx, to_idx = action
        from_pos = divmod(from_idx, self.cols)
        to_pos = divmod(to_idx, self.cols)

        reward = 0
        done = False

        prev_len = solver_path_length(self.maze)
    
        if self.maze[from_pos] == 1 and self.maze[to_pos] == 0:
            self.maze[from_pos] = 0
            self.maze[to_pos] = 1

            new_len = solver_path_length(self.maze)
            if new_len is None:
                reward = -10
                self.maze[from_pos] = 1
                self.maze[to_pos] = 0
            else:
                reward = new_len - prev_len  # positive if path got longer
        else:
            reward = -1


        return self.maze.flatten(), reward, done, {}

    def render(self, mode='human'):
        for row in self.maze:
            print("".join([' ' if c == 0 else '#' for c in row]))
        print()

    def _generate_solvable_maze(self):
        for _ in range(100):
            maze = generate_random_maze(self.rows, self.cols)
            maze[0, 0] = 0
            maze[self.rows - 1, self.cols - 1] = 0
            if is_solvable(maze):
                return maze
        raise RuntimeError("Failed to generate solvable maze")
