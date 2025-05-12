import gym
import numpy as np
from gym import spaces
from models.generator import generate_random_maze
from models.random_mutator import is_solvable

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

        # Only move if from_pos is wall and to_pos is path
        if self.maze[from_pos] == 1 and self.maze[to_pos] == 0:
            self.maze[from_pos] = 0
            self.maze[to_pos] = 1

            if is_solvable(self.maze):
                reward = 1
            else:
                reward = -10
                # Revert move
                self.maze[from_pos] = 1
                self.maze[to_pos] = 0
        else:
            reward = -1  # invalid move

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
