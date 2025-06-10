import numpy as np

from models.GNN_solver import GNNSolverAgent
from models.generator import generate_prim_algo

solver = GNNSolverAgent(lr=3e-4)

try:
    solver.load()
    print("Loaded model")
except FileNotFoundError:
    print("No saved model found")

max_epochs = 10000
batch_timesteps = 1024

for epoch in range(max_epochs):
    batch_data = {
        'graphs': [], 'log_probs': [], 'values': [], 'rewards': [], 'actions': [], 'entropies': [],
        'masks': [], 'dones': []
    }
    batch_returns = []
    episode_lengths = []
    timesteps = 0
    while timesteps < batch_timesteps:
        maze, start, goal = generate_prim_algo(7, 7)
        data = solver.run_episode(maze, start, goal)
        graphs, lp, vals, rews, acts, ents, masks, dones, path = data

        batch_data['graphs'].extend(graphs)
        batch_data['log_probs'].extend(lp)
        batch_data['values'].extend(vals)
        batch_data['rewards'].extend(rews)
        batch_data['actions'].extend(acts)
        batch_data['entropies'].extend(ents)
        batch_data['masks'].extend(masks)
        batch_data['dones'].extend(dones)

        batch_returns.append(sum(rews))
        episode_lengths.append(len(rews))
        timesteps += len(rews)

    actor_loss, critic_loss, entropy_loss = solver.update(
        batch_data['graphs'], batch_data['log_probs'], batch_data['values'],
        batch_data['rewards'], batch_data['actions'], batch_data['entropies'],
        batch_data['masks'], batch_data['dones']
    )

    avg_return = np.mean(batch_returns)
    avg_length = np.mean(episode_lengths)
    success_count = sum(1 for r in batch_returns if r >= 9.0)
    print(f"success_rate: {success_count}/{len(batch_returns)}")
    if success_count > 5:
        solver.save()

    print(f"Epoch {epoch:4d} | avg_return: {avg_return:.2f} | avg_len: {avg_length:.1f} | "
          f"actor_loss: {actor_loss:.3f} | critic_loss: {critic_loss:.3f} | entropy: {entropy_loss:.3f}")
    if epoch % 10 == 0:
        solver.save()