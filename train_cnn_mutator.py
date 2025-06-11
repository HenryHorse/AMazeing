import numpy as np
import torch
import matplotlib.pyplot as plt

from models.CNN_mutator import CNNUniqueMutatorAgent, shortest_path_length
from models.generator   import generate_prim_algo
from models.random_mutator import is_solvable

# ──────────────────────────── Hyperparameters ────────────────────────────
MAX_SIZE   = 23
BATCH_SIZE = 256
MAX_EPOCHS = 100000

# Curriculum: 200 epochs @ 7×7, 200 @ 11×11, 200 @ 15×15, rest @ 23×23
size_schedule = [5]*1000 + [7]*1000 + [11]*1000 + [15]*1000 + [19]*1000 + [MAX_SIZE]*(MAX_EPOCHS - 5000)
avg_rewards = []  # store avg_reward each epoch

# ─────────────────────── Agent Instantiation ───────────────────────
agent = CNNUniqueMutatorAgent(
    rows=MAX_SIZE,
    cols=MAX_SIZE,
    lr=3e-4,
    gamma=0.99,
    clip_eps=0.2,
    ppo_epochs=4,
    model_path="cnn_mutator_new.pt"
)

try:
    agent.load()
    print("✅ Loaded pretrained mutator")
except FileNotFoundError:
    print("🔄 No pretrained mutator found; starting from scratch")

# ─────────────────────────── Training Loop ───────────────────────────
for epoch in range(MAX_EPOCHS):
    size = size_schedule[epoch]  # current maze dimension
    batch_states, batch_actions, batch_oldlogps = [], [], []
    batch_values, batch_rewards, batch_ents, batch_dones = [], [], [], []

    for _ in range(BATCH_SIZE):
        # each "episode" is exactly one swap → reset unique‐swap history
        agent.reset()

        # 1) generate a solvable size×size maze
        while True:
            maze, start, goal = generate_prim_algo(size, size)
            if is_solvable(maze, start, goal):
                break

        # 2) compute baseline shortest‐path length
        base_len = shortest_path_length(maze, start, goal)

        # 3) agent picks & applies one mutation
        st, act, lp, val, ent = agent.select_action(maze, start, goal)
        new_maze, _, _, _      = agent.mutate(maze, start, goal)

        # 4) shaped reward: 
        new_len = shortest_path_length(new_maze, start, goal)
        if new_len is None:
            reward, done = -10.0, True
        else:
            delta = new_len - base_len
            if delta < -5: 
                delta = -5
            if delta > 0: 
                delta = delta * 2
            if delta == 0: 
                delta = 0.1
            reward = delta
            done   = True

        # 5) record the transition
        batch_states.append(st)
        batch_actions.append(torch.tensor(act))
        batch_oldlogps.append(lp)
        batch_values.append(val)
        batch_rewards.append(torch.tensor(reward))
        batch_ents.append(ent)
        batch_dones.append(done)

    # 6) PPO update on the whole batch
    agent.update(
        batch_states,
        batch_actions,
        batch_oldlogps,
        batch_values,
        batch_rewards,
        batch_ents,
        batch_dones
    )

    # 7) Logging
    avg_r = torch.stack(batch_rewards).float().mean().item()
    avg_rewards.append(avg_r)
    print(f"Epoch {epoch:4d} | size {size:2d}×{size:2d} | avg_reward {avg_r:+.3f}")

    # 8) Checkpoint
    if epoch % 10 == 0:
        agent.save()
        print("  ↪ Saved checkpoint")

        # Plot avg_reward curve so far
        plt.figure()
        plt.plot(avg_rewards, label="Avg Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title("PPO Mutator Training Progress")
        plt.legend()
        plt.savefig("mutator_training_curve.png")
        plt.close()
