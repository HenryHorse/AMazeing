from models.generator import generate_random_maze, generate_prim_algo
from models.random_mutator import MazeMutator, is_solvable
from visualize import run_mutating_visualization

def generate_solvable_maze(rows, cols, max_attempts=100):
    for _ in range(max_attempts):
        maze, goal = generate_prim_algo(rows, cols)
        if is_solvable(maze, start=(0, 0), goal=goal):
            return maze, goal
    raise RuntimeError("Could not generate a solvable maze after many attempts.")

def main():
    rows, cols = 10, 10
    mutator = MazeMutator()
    maze, goal = generate_solvable_maze(rows, cols)
    print("Initial maze is solvable. Starting live viewer...")
    run_mutating_visualization(maze, rows, cols, mutator, steps=20, interval_sec=3) # you should take in the goal_position for the maze as well here 


if __name__ == "__main__":
    main()