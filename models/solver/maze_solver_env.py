import gym
import numpy as np
from gym import spaces

class MazeSolverEnv(gym.Env):
    def __init__(self, maze, start, goal):
        super().__init__()
        self.maze = maze
        self.start = start
        self.goal = goal
        self.num_rows, self.num_cols = maze.shape

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_rows, self.num_cols), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.pos = self.start
        return self._get_obs()

    def _get_obs(self):
        obs = np.copy(self.maze).astype(np.float32)
        obs[self.pos] = 0.5
        obs[self.goal] = 0.9

    def step(self, action):
        row, col = self.pos
        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_row, new_col = row + delta[action][0], col + delta[action][1]

        if 0 <= new_row < self.num_rows and 0 <= new_col < self.num_cols and self.maze[new_row, new_col] == 0:
            self.pos = (new_row, new_col)

        done = self.pos == self.goal
        reward = 1.0 if done else -0.01
        return self._get_obs(), reward, done, {}
