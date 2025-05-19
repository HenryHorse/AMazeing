from models.generator import generate_random_maze, generate_maze_dfs_backtracker, prim_algo
from visualize import run_visualization

def main():
    num_rows, num_cols = 21, 21
    maze, start, goal = prim_algo(num_rows, num_cols)
    print(maze)
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    run_visualization(maze, num_rows, num_cols)

if __name__ == "__main__":
    main()