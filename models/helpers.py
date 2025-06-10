import numpy as np
from torch_geometric.data import HeteroData, Data
import torch



def get_valid_action_mask(maze, pos):
    r, c = pos
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    mask = []
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and maze[nr][nc] == 0:
            mask.append(True)
        else:
            mask.append(False)
    return torch.tensor(mask, dtype=torch.bool)

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def maze_to_tensor(maze, agent_pos, start_pos, goal_pos):
    rows, cols = maze.shape
    tensor = np.zeros((4, rows, cols), dtype=np.float32)
    tensor[0] = maze
    tensor[1][agent_pos] = 1.0
    tensor[2][start_pos] = 1.0
    tensor[3][goal_pos] = 1.0

    return torch.tensor(tensor).unsqueeze(0)

def maze_to_hetero_graph(maze, agent_pos):
    rows, cols = maze.shape
    cell_nodes = []
    wall_nodes = []
    cell_map = {}
    wall_map = {}

    cell_idx = 0
    wall_idx = 0

    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:
                cell_map[(r, c)] = cell_idx
                cell_nodes.append([1.0 if (r, c) == agent_pos else 0.0])
                cell_idx += 1
            else:
                wall_map[(r, c)] = wall_idx
                wall_nodes.append([0.0])
                wall_idx += 1

    data = HeteroData()
    data['cell'].x = torch.tensor(cell_nodes, dtype=torch.float32)
    data['wall'].x = torch.tensor(wall_nodes, dtype=torch.float32)

    cell_edges = []
    for (r, c), i in cell_map.items():
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_map:
                cell_edges.append([i, cell_map[(nr, nc)]])
    if cell_edges:
        data['cell', 'to', 'cell'].edge_index = torch.tensor(cell_edges, dtype=torch.long).t().contiguous()

    wall_cell_edges = []
    for (r, c), w_i in wall_map.items():
        for dr, dc in [(-1,0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_map:
                wall_cell_edges.append([w_i, cell_map[(nr, nc)]])
    if wall_cell_edges:
        data['wall', 'to', 'cell'].edge_index = torch.tensor(wall_cell_edges, dtype=torch.long).t().contiguous()


    return data


def maze_to_homogeneous_graph(maze, agent_pos, start_pos=None, goal_pos=None):
    rows, cols = maze.shape
    features = []
    edge_index = []

    def flat_idx(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            is_wall = float(maze[r][c] == 1)
            is_agent_here = float((r, c) == agent_pos)
            is_start = float((r, c) == start_pos) if start_pos else 0.0
            is_goal = float((r, c) == goal_pos) if goal_pos else 0.0
            features.append([
                is_wall,
                is_agent_here,
                is_start,
                is_goal,
                r / rows,
                c / cols,
            ])

    for r in range(rows):
        for c in range(cols):
            u = flat_idx(r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    v = flat_idx(nr, nc)
                    edge_index.append([u, v])

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T.contiguous()

    return Data(x=x, edge_index=edge_index)