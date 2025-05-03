from models.generator import generate_random_maze
from visualize import run_visualization

def main():
    num_rows, num_cols = 10, 10
    maze = generate_random_maze(num_rows, num_cols)
    print(maze)
    run_visualization(maze, num_rows, num_cols)

if __name__ == "__main__":
    main()