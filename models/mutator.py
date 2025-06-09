import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.random_mutator import is_solvable
import torch_geometric.nn as gnn
from torch_geometric.data import HeteroData, Data
from helpers import maze_to_homogeneous_graph

import torch
torch.autograd.set_detect_anomaly(True)



class GNNMutatorPolicy(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.gcn1 = gnn.GCNConv(input_dim, hidden_dim)
        self.gcn2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.open_head = nn.Linear(hidden_dim, 1)
        self.wall_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = F.relu(self.gcn1(data.x, data.edge_index))
        x = F.relu(self.gcn2(x, data.edge_index)).clone()  # .clone() helps avoid in-place issues

        open_logits = self.open_head(x).squeeze(-1)
        wall_logits = self.wall_head(x).squeeze(-1)

        return open_logits, wall_logits



class MazeMutatorAgent:
    def __init__(self, lr=1e-3, model_path="mutator.pt"):
        self.policy = GNNMutatorPolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.model_path = model_path

    def save(self):
        torch.save(self.policy.state_dict(), self.model_path)

    def load(self):
        self.policy.load_state_dict(torch.load(self.model_path))
        self.policy.eval()

    def get_candidates(self, maze):
        wall_positions = list(zip(*np.where(maze == 1)))
        path_positions = list(zip(*np.where(maze == 0)))
        candidates = []

        for wall in wall_positions:
            for path in path_positions:
                if wall != path:
                    candidates.append((wall, path))

        return candidates

    def mutate(self, maze, action):
        (from_pos, to_pos) = action
        maze = maze.copy()
        maze[from_pos] = 0
        maze[to_pos] = 1
        return maze

    def compute_discounted_returns(self, rewards, gamma=0.99):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        return returns

    def select_action(self, maze, agent_pos, start_pos, goal_pos):
        graph = maze_to_homogeneous_graph(maze, agent_pos, start_pos, goal_pos)
        open_logits, wall_logits = self.policy(graph)

        candidates = self.get_candidates(maze)
        if not candidates:
            raise ValueError("No valid (wall, open) swap candidates found")

        rows, cols = maze.shape
        def flatten(pos): return pos[0] * cols + pos[1]

        # Convert (row, col) → flat index to index into logits
        open_indices = torch.tensor([flatten(open_pos) for _, open_pos in candidates])
        wall_indices = torch.tensor([flatten(wall_pos) for wall_pos, _ in candidates])

        # Grab logits for each candidate
        open_scores = open_logits[open_indices]
        wall_scores = wall_logits[wall_indices]
        pair_scores = open_scores + wall_scores

        dist = torch.distributions.Categorical(logits=pair_scores)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)

        return candidates[idx.item()], log_prob

    def safe_mutate(self, maze, action, start, goal):
        mutated = self.mutate(maze, action)
        if is_solvable(mutated, start, goal):
            return mutated
        return maze  # Or try another candidate, depending on setup


    def update(self, log_probs, rewards):
        self.optimizer.zero_grad()

        if not log_probs:
            return
        returns = self.compute_discounted_returns(rewards)
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G
        loss.backward()
        self.optimizer.step()

    

 
