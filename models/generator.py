import numpy as np

# baseline for visualizations, not actual mazes
def generate_random_maze(num_rows, num_cols):
    return np.random.randint(0, 2, size=(num_rows, num_cols))
