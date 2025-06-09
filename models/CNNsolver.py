import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
import numpy as np
from helpers import maze_to_homogeneous_graph, manhattan_distance, maze_to_tensor
from generator import generate_prim_algo



class CNNSolverPolicy(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        self.policy_head = nn.Linear(hidden_dim, 4)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        agent_mask = x[:, 1:2]
        agent_yx = (agent_mask.squeeze(1) == agent_mask.max()).nonzero()[0][-2:]
        agent_feature = x[0, :, agent_yx[0], agent_yx[1]]

        logits = self.policy_head(agent_feature).squeeze(0)
        value = self.value_head(agent_feature).squeeze(0)
        return logits, value


class CNNSolverAgent:
    def __init__(self, lr=3e-4, gamma=0.99, clip_eps=0.2, ppo_epochs=10, model_path="cnn_solver.pt"):
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

    def select_action(self, maze_tensor, agent_pos):
        logits, value = self.policy(maze_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy

    def run_episode(self, maze, start, goal, max_steps=500):
        pos = start
        tensors, log_probs, values, rewards, actions, entropies, path = [], [], [], [], [], [], [pos]

        for _ in range(max_steps):
            maze_tensor = maze_to_tensor(maze, pos, start, goal)
            with torch.no_grad():
                action, log_prob, value, entropy = self.select_action(maze_tensor, pos)

            tensors.append(maze_tensor)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            actions.append(action)

            if pos == goal:
                rewards.append(10.0)
                break


            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action.item()]
            nr, nc = pos[0] + dr, pos[1] + dc
            dist_before = manhattan_distance(pos, goal)

            if (0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and maze[nr][nc] == 0):
                pos = (nr, nc)
                path.append(pos)
                dist_after = manhattan_distance(pos, goal)
                shaping = 2 * (dist_before - dist_after)
                rewards.append(-0.1 + shaping)
            else:
                rewards.append(-0.5)

        return tensors, log_probs, values, rewards, actions, entropies, path


    def compute_returns_and_advantages(self, rewards, values):
        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        values = torch.stack(values).squeeze(-1)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return returns, advantages


    def update(self, tensors, log_probs, values, rewards, actions, entropies):
        old_log_probs = torch.stack(log_probs).detach()
        returns, advantages = self.compute_returns_and_advantages(rewards, values)


        for _ in range(self.ppo_epochs):
            new_log_probs, new_values, new_entropies = [], [], []
            for tensor, action in zip(tensors, actions):
                logits, value = self.policy(tensor)
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
        logits, _ = self.policy(maze_tensor)
        return torch.argmax(F.softmax(logits, dim=-1), dim=-1).item()



# maze = np.array([
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0],
# ])
# start = (0, 0)
# goal = (2, 2)

solver = CNNSolverAgent(lr=3e-4)

try:
    solver.load()
    print("Loaded model")
except FileNotFoundError:
    print("No saved model found")

max_epochs = 10000
batch_timesteps = 1024

for epoch in range(max_epochs):
    batch_data = {
        'tensors': [], 'log_probs': [], 'values': [], 'rewards': [], 'actions': [], 'entropies': []
    }
    batch_returns = []
    episode_lengths = []
    timesteps = 0
    while timesteps < batch_timesteps:
        maze, start, goal = generate_prim_algo(7, 7)
        data = solver.run_episode(maze, start, goal)
        tensors, lp, vals, rews, acts, ents, path = data

        batch_data['tensors'].extend(tensors)
        batch_data['log_probs'].extend(lp)
        batch_data['values'].extend(vals)
        batch_data['rewards'].extend(rews)
        batch_data['actions'].extend(acts)
        batch_data['entropies'].extend(ents)

        batch_returns.append(sum(rews))
        episode_lengths.append(len(rews))
        timesteps += len(rews)

    actor_loss, critic_loss, entropy_loss = solver.update(
        batch_data['tensors'], batch_data['log_probs'], batch_data['values'],
        batch_data['rewards'], batch_data['actions'], batch_data['entropies']
    )

    avg_return = np.mean(batch_returns)
    avg_length = np.mean(episode_lengths)
    success_count = sum(1 for r in batch_returns if r >= 9.0)
    print(f"success_rate: {success_count}/{len(batch_returns)}")

    print(f"Epoch {epoch:4d} | avg_return: {avg_return:.2f} | avg_len: {avg_length:.1f} | "
          f"actor_loss: {actor_loss:.3f} | critic_loss: {critic_loss:.3f} | entropy: {entropy_loss:.3f}")
    if epoch % 10 == 0:
        solver.save()