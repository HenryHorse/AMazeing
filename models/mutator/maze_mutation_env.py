import gym
import numpy as np
from gym import spaces
from models.generator import generate_random_maze
from models.random_mutator import is_solvable
import networkx as nx


from queue import Queue

def is_solvable(maze, start, goal):
    rows, cols = maze.shape
    visited = np.zeros_like(maze)
    q = Queue()
    q.put(start)
    visited[start] = 1

    while not q.empty():
        r, c = q.get()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr, nc] == 0 and visited[nr, nc] == 0:
                    visited[nr, nc] = 1
                    q.put((nr, nc))
    return False


class MazeMutatorEnv(gym.Env):
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.rows, self.cols = maze.shape
        self.action_space = spaces.Discrete(self.rows * self.cols)
        self.observation_space = spaces.Box(low=0, high=1, shape=maze.shape, dtype=np.uint8)

    def reset(self):
        self.done = False
        return self.maze.copy()

    def step(self, action):
        r, c = divmod(action, self.cols)
        if (r, c) == self.start or (r, c) == self.goal:
            reward = -10
            self.done = True
            return self.maze.copy(), reward, self.done, {}
        
        # toggle the cell: path <-> wall
        self.maze[r, c] = 1 - self.maze[r, c]
        if not is_solvable(self.maze, self.start, self.goal):
            reward = -10
        else:
            reward = 0  # actual reward comes from later evaluation
        self.done = True
        return self.maze.copy(), reward, self.done, {}
