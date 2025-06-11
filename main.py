from models.CNN_solver import CNNSolverAgent
from models.GNN_solver import GNNSolverAgent
from models.generator import generate_random_maze, generate_prim_algo
from models.random_mutator import MazeMutator, is_solvable
from visualize import run_mutating_visualization, run_visualization, solver_visualization, solver_and_mutator_visualization, Astar_and_mutator_visualization, Astar_and_PPOmutator_visualization
from models.heuristic_mutator import SimpleHeuristicMutator, TrueRandomMutator
from models.CNN_mutator import CNNUniqueMutatorAgent

def generate_solvable_maze(rows, cols, max_attempts=100):
    for _ in range(max_attempts):
        maze, start, goal = generate_prim_algo(rows, cols)
        if is_solvable(maze, start=start, goal=goal):
            return maze, start, goal
    raise RuntimeError("Could not generate a solvable maze after many attempts.")

def main():
    rows, cols = 23, 23
    mutator = TrueRandomMutator()
    #mutator = CNNUniqueMutatorAgent(rows, cols, model_path="cnn_mutator_new.pt")
    #try:
    #    mutator.load()
    #    print("Loaded trained PPO mutator.")
    #except FileNotFoundError:
    #    print("No pretrained mutator found; running with random initialization.")


    solver = GNNSolverAgent()

    maze, start, goal = generate_solvable_maze(rows, cols)
    print("Initial maze is solvable. Starting live viewer...")
    #run_mutating_visualization(maze, rows, cols, mutator, start, goal, steps=20, interval_sec=3) # you should take in the goal_position for the maze as well here
    solved = False
    while not solved:
        maze, start, goal = generate_solvable_maze(rows, cols)
        solved = Astar_and_mutator_visualization(mutator, maze, start, goal)


if __name__ == "__main__":
    main()