
# convert to graph
import networkx as nx
import numpy as np

def maze_to_graph(maze):
    G = nx.Graph()
    rows, cols = maze.shape

    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:  # only consider free cells
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4 directions
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                        G.add_edge((r, c), (nr, nc))
    return G


def graph_to_maze(G, shape):
    maze = np.ones(shape, dtype=int)
    for r, c in G.nodes:
        maze[r][c] = 0
    return maze


def is_solvable(maze, start=(0, 0), goal=None):
    G = maze_to_graph(maze)
    if goal is None:
        goal = (maze.shape[0] - 1, maze.shape[1] - 1)

    # If either start or goal isn't walkable, maze is unsolvable
    if start not in G or goal not in G:
        return False

    return nx.has_path(G, start, goal)


class MazeMutator:
    def __init__(self):
        pass

    def mutate(self, maze, timestep):
        print("Making a change!")
        maze = maze.copy()
        rows, cols = maze.shape

        wall_positions = list(zip(*np.where(maze == 1)))
        path_positions = list(zip(*np.where(maze == 0)))

        np.random.shuffle(wall_positions)
        np.random.shuffle(path_positions)

        for from_pos in wall_positions:
            for to_pos in path_positions:
                maze[from_pos] = 0
                maze[to_pos] = 1
                if is_solvable(maze):
                    return maze  # success!
                else:
                    maze[from_pos] = 1
                    maze[to_pos] = 0

        print("No valid wall move found.")
        return maze  # return unchanged if no move worked
