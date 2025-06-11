import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import deque
from models.helpers import maze_to_tensor
from models.random_mutator import is_solvable


def shortest_path_length(maze: np.ndarray, start: tuple, goal: tuple) -> int | None:
    """
    BFS to compute shortest path length (number of steps) from start to goal,
    or None if unsolvable.
    """
    rows, cols = maze.shape
    visited = set([start])
    queue = deque([(start, 0)])
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    while queue:
        (r,c), dist = queue.popleft()
        if (r,c) == goal:
            return dist
        for dr, dc in moves:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr,nc]==0 and (nr,nc) not in visited:
                visited.add((nr,nc))
                queue.append(((nr,nc), dist+1))
    return None


class CNNUniqueMutatorPolicy(nn.Module):
    """
    CNN → global‐average → separate heads over removal and addition indices.
    """
    def __init__(self, rows: int, cols: int, input_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.cell_count = rows * cols

        # convolutional backbone
        self.conv1 = nn.Conv2d(input_dim,  hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        # two heads: one for which wall to remove, one for where to add
        self.remove_head = nn.Linear(hidden_dim, self.cell_count)  # logits over all cells
        self.add_head    = nn.Linear(hidden_dim, self.cell_count)  # logits over all cells

        # value head for critic
        self.value_head  = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 4, H, W] tensor
        returns:
          remove_logits: [B, H*W]
          add_logits:    [B, H*W]
          values:        [B]
        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        # global average pooling
        feat = h.mean(dim=[2, 3])  # shape [B, hidden_dim]

        # heads
        remove_logits = self.remove_head(feat)  # [B, cell_count]
        add_logits    = self.add_head(feat)     # [B, cell_count]
        values        = self.value_head(feat).squeeze(-1)  # [B]

        return remove_logits, add_logits, values


class CNNUniqueMutatorAgent:
    """
    PPO agent that picks a single (remove, add) pair each mutate(),
    tracks seen swaps to mask them, and is trained to maximize Δ path‐length.
    """
    def __init__(
        self,
        rows: int,
        cols: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        model_path: str = "cnn_mutator_new.pt",
    ):
        self.rows, self.cols = rows, cols
        self.cell_count = rows * cols
        self.policy = CNNUniqueMutatorPolicy(rows, cols)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma    = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.model_path = model_path

        # track which (remove,add) pairs we've done this episode
        self.seen_swaps = set()

    def reset(self):
        """Clear out seen swap history at episode start."""
        self.seen_swaps.clear()

    def save(self):
        torch.save(self.policy.state_dict(), self.model_path)

    def load(self):
        self.policy.load_state_dict(torch.load(self.model_path))
        self.policy.eval()
    def select_action(self, maze, solver_pos, goal):
        """
        Sample one (remove,add) swap from a two-headed policy,
        masking out out‐of‐bounds cells for smaller mazes and seen‐swaps.
        Returns: state, (r_act,a_act), logp, value, entropy
        """
        # 1) Build state tensor [1,4,H,W]
        state = maze_to_tensor(maze, solver_pos, solver_pos, goal)

        # 2) Get raw logits & value
        remove_logits, add_logits, value = self.policy(state)  # [1,C], [1,C], [1]
        remove_logits = remove_logits.view(-1).clone()          # [C]
        add_logits    = add_logits.view(-1).clone()             # [C]

        # 3) Small-grid masks from actual maze size
        H, W = maze.shape
        small_count = H * W

        m = state[0,0]  # maze channel [H,W]
        s = state[0,2]  # solver channel
        g = state[0,3]  # goal channel

        small_wall_mask  = (m == 1).view(-1)       # can only remove existing walls
        small_empty_mask = (m == 0).view(-1)       # can only add on empty
        small_empty_mask &= (s == 0).view(-1)      # never add at solver
        small_empty_mask &= (g == 0).view(-1)      # never add at goal

        # 4) Pad those masks to full 23*23 = C entries
        C = self.cell_count
        wall_mask  = torch.zeros(C, dtype=torch.bool)
        empty_mask = torch.zeros(C, dtype=torch.bool)
        wall_mask[:small_count]  = small_wall_mask
        empty_mask[:small_count] = small_empty_mask

        # 5) Exclude any swaps already in seen_swaps
        for (rem, add) in self.seen_swaps:
            rem_idx = rem[0]*self.cols + rem[1]
            add_idx = add[0]*self.cols + add[1]
            wall_mask[rem_idx]   = False
            empty_mask[add_idx]  = False

        # 6) Mask out invalid logits
        remove_logits[~wall_mask] = -1e9
        add_logits[~empty_mask]   = -1e9

        # 7) Sample actions as scalar tensors
        dist_r = torch.distributions.Categorical(logits=remove_logits)
        dist_a = torch.distributions.Categorical(logits=add_logits)
        r_act_tensor = dist_r.sample()   # tensor scalar
        a_act_tensor = dist_a.sample()

        # 8) Compute joint log‐prob & entropy
        logp = dist_r.log_prob(r_act_tensor) + dist_a.log_prob(a_act_tensor)
        ent  = dist_r.entropy() + dist_a.entropy()

        # 9) Convert to Python ints
        r_act = int(r_act_tensor.item())
        a_act = int(a_act_tensor.item())

        return state, (r_act, a_act), logp, value, ent

    def mutate(self, maze, solver_pos, goal):
        # sample one swap
        state, (r_idx, a_idx), logp, value, ent = self.select_action(maze, solver_pos, goal)

        # use the local maze shape, not self.cols
        H, W = maze.shape
        rr, rc = divmod(r_idx, W)
        ar, ac = divmod(a_idx, W)

        # apply it
        new_maze = maze.copy()
        new_maze[rr, rc] = 0
        new_maze[ar, ac] = 1

        # record for unique‐swap constraint
        self.seen_swaps.add(((rr, rc), (ar, ac)))

        return new_maze, logp, value, ent


    def compute_returns_and_advantages(self, rewards, values, dones):
        returns = []
        G = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        vals    = torch.stack(values).squeeze(-1)
        advs    = returns - vals.detach()
        advs    = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)
        return returns, advs

    def update(self, states, actions, old_logps, values, rewards, entropies, dones):
        returns, advs = self.compute_returns_and_advantages(rewards, values, dones)
        old_logps = torch.stack(old_logps).detach()

        for _ in range(self.ppo_epochs):
            new_logps, new_vals, new_ents = [], [], []

            for st, (r_act, a_act) in zip(states, actions):
                # 1) get fresh logits & value
                r_logits, a_logits, val = self.policy(st)   # [1,C], [1,C], [1]
                r_logits = r_logits.view(-1).clone()        # [C]
                a_logits = a_logits.view(-1).clone()        # [C]

                # 2) rebuild the small masks from the actual maze size
                H, W = st.shape[2], st.shape[3]
                small_count = H * W
                m = st[0,0]  # maze channel
                s = st[0,2]  # solver channel
                g = st[0,3]  # goal channel

                small_wall_mask  = (m == 1).view(-1)
                small_empty_mask = (m == 0).view(-1)
                small_empty_mask &= (s == 0).view(-1)
                small_empty_mask &= (g == 0).view(-1)

                # 3) pad those masks to full C = 23*23
                C = self.cell_count
                wall_mask  = torch.zeros(C, dtype=torch.bool)
                empty_mask = torch.zeros(C, dtype=torch.bool)
                wall_mask[:small_count]  = small_wall_mask
                empty_mask[:small_count] = small_empty_mask

                # 4) exclude any seen swaps
                for (rem, add) in self.seen_swaps:
                    rem_idx = rem[0]*self.cols + rem[1]
                    add_idx = add[0]*self.cols + add[1]
                    wall_mask[rem_idx]  = False
                    empty_mask[add_idx] = False

                # 5) mask out invalid logits
                r_logits[~wall_mask] = -1e9
                a_logits[~empty_mask] = -1e9

                # 6) recompute log‐prob & entropy for the taken action
                dist_r = torch.distributions.Categorical(logits=r_logits)
                dist_a = torch.distributions.Categorical(logits=a_logits)
                new_logp = dist_r.log_prob(torch.tensor(r_act)) + dist_a.log_prob(torch.tensor(a_act))
                new_ent  = dist_r.entropy() + dist_a.entropy()

                new_logps.append(new_logp)
                new_vals.append(val.squeeze(-1))
                new_ents.append(new_ent)

            # stack and take PPO step as before
            new_logps = torch.stack(new_logps)
            new_vals  = torch.stack(new_vals)
            new_ents  = torch.stack(new_ents)

            ratio = torch.exp(new_logps - old_logps)
            s1 = ratio * advs
            s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs

            actor_loss  = -torch.min(s1, s2).mean()
            critic_loss = F.mse_loss(new_vals, returns)
            entropy_loss= -new_ents.mean()

            self.optimizer.zero_grad()
            (actor_loss + 0.5*critic_loss + 0.01*entropy_loss).backward()
            self.optimizer.step()
