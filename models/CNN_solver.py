from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
import numpy as np
from models.helpers import maze_to_homogeneous_graph, manhattan_distance, maze_to_tensor, get_valid_action_mask
from models.generator import generate_prim_algo



class CNNSolverPolicy(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        self.policy_head = nn.Linear(hidden_dim, 4)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, agent_pos):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        r, c = agent_pos
        agent_feature = x[0, :, r, c]

        logits = self.policy_head(agent_feature).squeeze(0)
        value = self.value_head(agent_feature).squeeze(0)
        return logits, value


class CNNSolverAgent:
    def __init__(self, lr=3e-4, gamma=0.99, clip_eps=0.2, ppo_epochs=10, model_path="cnn_solver_new.pt"):
        self.policy = CNNSolverPolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.model_path = model_path

    def save(self):
        torch.save(self.policy.state_dict(), self.model_path)

    def load(self):
        self.policy.load_state_dict(torch.load(self.model_path))
        self.policy.eval()

    def select_action(self, maze_tensor, agent_pos, maze):
        logits, value = self.policy(maze_tensor, agent_pos)
        valid_mask = get_valid_action_mask(maze, agent_pos)
        logits[~valid_mask] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy, valid_mask

    def run_episode(self, maze, start, goal, max_steps=500):
        pos = start
        visited = set([pos])
        visit_count = defaultdict(int)
        tensors, log_probs, values, rewards = [], [], [], []
        actions, entropies, agent_positions, masks, dones = [], [], [], [], []
        path = [pos]

        for _ in range(max_steps):
            maze_tensor = maze_to_tensor(maze, pos, start, goal)
            with torch.no_grad():
                action, log_prob, value, entropy, valid_mask = self.select_action(maze_tensor, pos, maze)

            tensors.append(maze_tensor)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            actions.append(action)
            agent_positions.append(pos)
            masks.append(valid_mask)

            if pos == goal:
                rewards.append(10.0)
                dones.append(True)
                break

            dist_before = manhattan_distance(pos, goal)
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action.item()]
            pos = (pos[0] + dr, pos[1] + dc)
            path.append(pos)

            visit_count[pos] += 1
            if visit_count[pos] > 1:
                r = -0.1 * visit_count[pos]
            else:
                dist_after = manhattan_distance(pos, goal)
                r = -0.1 + 2 * (dist_before - dist_after)
            rewards.append(r)
            dones.append(False)

        if len(rewards) == max_steps and pos != goal:
            rewards[-1] -= 10
            dones[-1] = True

        return tensors, log_probs, values, rewards, actions, entropies, agent_positions, masks, dones, path

    def compute_returns_and_advantages(self, rewards, values, dones):
        returns = torch.zeros(len(rewards))
        G = 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                G = 0.0
            G = rewards[i] + self.gamma * G
            returns[i] = G

        returns = returns.clamp(-20.0, 20.0)

        values_tensor = torch.stack(values).squeeze(-1)
        advantages = returns - values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return returns, advantages


    def update(self, tensors, log_probs, values, rewards, actions, entropies, agent_positions, masks, dones):
        old_log_probs = torch.stack(log_probs).detach()
        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)


        for _ in range(self.ppo_epochs):
            new_log_probs, new_values, new_entropies = [], [], []
            for tensor, action, pos, mask in zip(tensors, actions, agent_positions, masks):
                logits, value = self.policy(tensor, pos)
                logits[~mask] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs.append(dist.log_prob(action))
                new_values.append(value.squeeze(-1))
                new_entropies.append(dist.entropy())

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)
            new_entropies = torch.stack(new_entropies)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, returns)
            entropy_loss = -new_entropies.mean()

            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy_loss.item()


    def get_next_move(self, maze, pos, start, goal):
        maze_tensor = maze_to_tensor(maze, pos, start, goal)
        logits, _ = self.policy(maze_tensor, pos)
        valid_mask = get_valid_action_mask(maze, pos)
        logits[~valid_mask] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action


