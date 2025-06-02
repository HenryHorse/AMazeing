from stable_baselines3 import PPO
from models.mutator.maze_mutation_env import MazeMutationEnv

env = MazeMutationEnv(rows=10, cols=10)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)


