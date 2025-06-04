class MazeMutatorPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, maze_tensor):
        return self.net(maze_tensor)

class MazeMutatorAgent:
    def __init__(self, shape=(10, 10), lr=1e-3):
        self.input_dim = shape[0] * shape[1]
        self.policy = MazeMutatorPolicy(self.input_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, maze):
        flat = torch.tensor(maze.flatten(), dtype=torch.float32)
        logits = self.policy(flat)
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def mutate(self, maze, action):
        maze = maze.copy()
        r, c = divmod(action, maze.shape[1])
        maze[r, c] = 1 - maze[r, c]
        return maze

    def update(self, log_probs, rewards):
        returns = torch.tensor(rewards, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
