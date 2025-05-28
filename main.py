from models.generator import generate_random_maze
from models.random_mutator import MazeMutator, is_solvable
from visualize import run_mutating_visualization

def generate_solvable_maze(rows, cols, max_attempts=100):
    for _ in range(max_attempts):
        maze = generate_random_maze(rows, cols)
        if is_solvable(maze, start=(0, 0), goal=(rows - 1, cols - 1)):
            return maze
    raise RuntimeError("Could not generate a solvable maze after many attempts.")

def main():
    rows, cols = 10, 10
    mutator = MazeMutator()
    maze = generate_solvable_maze(rows, cols)
    print("Initial maze is solvable. Starting live viewer...")
    run_mutating_visualization(maze, rows, cols, mutator, steps=20, interval_sec=3)


if __name__ == "__main__":
    main()