import torch
from torch_geometric.data import HeteroData, Data
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
import numpy as np




# New maze_to_hetero function that includes the end_position as well into the graph.
# added is_end_here feature to the node's vector. Simply 2 lines added to old maze-to-hetero function.
def maze_to_homogeneous_graph(maze, agent_pos, end_pos):
    rows, cols = maze.shape
    num_nodes = rows * cols

    features = []
    edge_index = []

    def flat_idx(row, col):
        return row * cols + col

    for r in range(rows):
        for c in range(cols):
            is_wall = float(maze[r][c] == 1)
            is_agent_here = float((r, c) == agent_pos)
            is_end_here = float((r, c) == end_pos)      # this line
            features.append([
                is_wall,
                is_agent_here,
                is_end_here,         # and this line
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
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

# the old version of the converter function without end_position taken into account.
def old_maze_to_homogeneous_graph(maze, agent_pos, start_pos=None, goal_pos=None):
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



class GNNSolverPolicy(nn.Module):
    def __init__(self, input_dim = 6, hidden_dim=64):
        super().__init__()
        # self.convs = gnn.HeteroConv({
        #     ('cell', 'to', 'cell'): gnn.GATConv(1, hidden_dim, add_self_loops=False),
        #     ('wall', 'to', 'cell'): gnn.GATConv(1, hidden_dim, add_self_loops=False),
        # }, aggr='sum')
        self.gcn1 = gnn.GCNConv(input_dim, hidden_dim)
        self.gcn2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, 4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))

        agent_idx = (data.x[:, 1] == 1.0).nonzero(as_tuple=True)[0]

        agent_embed = x[agent_idx]

        # x_dict = self.convs(data.x_dict, data.edge_index_dict)
        # agent_idx = (data['cell'].x[:, 0] == 1.0).nonzero(as_tuple=True)[0]
        # agent_embed = x_dict['cell'][agent_idx]

        return self.policy_head(agent_embed).squeeze(0)


class GNNSolverAgent:
    def __init__(self, lr=1e-3, gamma=0.99, model_path="solver.pt"):
        self.policy = GNNSolverPolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.model_path = model_path

    def save(self):
        torch.save(self.policy.state_dict(), self.model_path)

    def load(self):
        self.policy.load_state_dict(torch.load(self.model_path))
        self.policy.eval()

    def select_action(self, maze, pos, start, goal):
        graph = maze_to_homogeneous_graph(maze, pos, start, goal)
        logits = self.policy(graph)
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def run_episode(self, maze, start, goal, max_steps=100):
        pos = start
        log_probs = []
        rewards = []
        path = [pos]

        for _ in range(max_steps):
            if pos == goal:
                rewards.append(10.0)
                break
            action, log_prob = self.select_action(maze, pos, start, goal)
            log_probs.append(log_prob)

            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            nr, nc = pos[0] + dr, pos[1] + dc

            if (0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and maze[nr][nc] == 0):
                pos = (nr, nc)
                path.append(pos)
                rewards.append(-0.01)
            else:
                rewards.append(-1.0)
                break

        return log_probs, rewards, path


    def compute_discounted_returns(self, rewards):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        return returns


    def update(self, log_probs, rewards):
        returns = self.compute_discounted_returns(rewards)
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_next_move(self, maze, pos, start, goal):
        action, _ = self.select_action(maze, pos, start, goal) # we need to fix this too I think
        return action



maze = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])
start = (0, 0)
goal = (2, 2)

# homogeneous_graph = maze_to_homogeneous_graph(maze, start)
# print(homogeneous_graph.x)
# print(homogeneous_graph.edge_index)
solver = GNNSolverAgent(lr=0.01)

try:
    solver.load()
    print("Loaded model")
except FileNotFoundError:
    print("No saved model found")

for epoch in range(1000):
    log_probs, rewards, path = solver.run_episode(maze, start, goal)
    solver.update(log_probs, rewards)
    if path[-1] == goal:
        print(f"Reached goal in {len(path)} steps at epoch {epoch}")
        print("Path:", path)
        solver.save()
        break