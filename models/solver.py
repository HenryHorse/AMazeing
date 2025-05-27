import torch
from torch_geometric.data import HeteroData
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
import numpy as np


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
        for dr, dc in [(-1,0), (1, 0), (0, -1), (0, 1)]:
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



class GNNSolverPolicy(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.convs = gnn.HeteroConv({
            ('cell', 'to', 'cell'): gnn.GATConv(1, hidden_dim, add_self_loops=False),
            ('wall', 'to', 'cell'): gnn.GATConv(1, hidden_dim, add_self_loops=False),
        }, aggr='sum')
        self.policy_head = nn.Linear(hidden_dim, 4)

    def forward(self, data):
        x_dict = self.convs(data.x_dict, data.edge_index_dict)

        agent_idx = (data['cell'].x[:, 0] == 1.0).nonzero(as_tuple=True)[0]
        agent_embed = x_dict['cell'][agent_idx]

        return self.policy_head(agent_embed).squeeze(0)


class GNNSolverAgent:
    def __init__(self, lr=1e-3):
        self.policy = GNNSolverPolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, maze, pos):
        graph = maze_to_hetero_graph(maze, pos)
        logits = self.policy(graph)
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, log_prob, reward):
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




maze = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])
start = (0, 0)
goal = (2, 2)

solver = GNNSolverAgent(lr=0.01)

