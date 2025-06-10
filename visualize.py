import pygame
import time
from models.random_mutator import maze_to_graph
import networkx as nx

def draw_maze(screen, maze, rows, cols):
    width, height = screen.get_size()

    cell_size = min(width // cols, height // rows)

    screen.fill((0, 0, 0))

    for row in range(rows):
        for col in range(cols):
            color = (255, 255, 255) if maze[row][col] == 0 else (0, 0, 0)
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)


def run_visualization(maze, rows, cols):
    pygame.init()

    info = pygame.display.Info()

    screen_width = int(info.current_w * 0.5)
    screen_height = int(info.current_h * 0.5)

    min_dimension = min(screen_width, screen_height)
    screen = pygame.display.set_mode((min_dimension, min_dimension))
    pygame.display.set_caption("Maze Visualization")

    running = True
    while running:
        draw_maze(screen, maze, rows, cols)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


def run_mutating_visualization(initial_maze, rows, cols, mutator, steps=20, interval_sec=3): 
    pygame.init()

    screen_width = 600
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Live Maze Visualization")

    maze = initial_maze.copy()
    clock = pygame.time.Clock()
    last_update_time = time.time()
    t = 0

    running = True
    while running and t < steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        current_time = time.time()
        if current_time - last_update_time >= interval_sec:
            print(f"Timestep {t}: Mutating...")
            maze = mutator.mutate(maze, timestep=t)
            t += 1
            last_update_time = current_time

        draw_maze(screen, maze, rows, cols)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def solver_visualization(solver, maze, start, goal):
    pygame.init()
    rows, cols = maze.shape
    screen_size = 600
    cell_size = screen_size // max(rows, cols)
    screen = pygame.display.set_mode((cols * cell_size, rows * cell_size))
    pygame.display.set_caption("Agent Solving Maze")

    clock = pygame.time.Clock()
    running = True
    pos = start
    path = [pos]
    reached_goal = False

    try:
        G = maze_to_graph(maze)
        astar_path = nx.astar_path(G, start, goal, heuristic=lambda a, b: abs(a[0]-b[0] + abs(a[1]-b[1])))
    except Exception as e:
        print("A* failed", e)
        astar_path = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_maze(screen, maze, rows, cols)


        for r, c in astar_path:
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 255, 255), rect)

        for i, (r, c) in enumerate(path):
            intensity = int(100 + 155 * (i / len(path)))
            color = (intensity, 0, 0)
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)

        goal_color = (0, 0, 255) if reached_goal else (0, 255, 0)
        goal_rect = pygame.Rect(goal[1] * cell_size, goal[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, goal_color, goal_rect)

        pygame.display.flip()
        time.sleep(0.1)

        if pos == goal:
            reached_goal = True
            continue

        next_action = solver.get_next_move(maze, pos, start, goal)
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][next_action]
        nr, nc = pos[0] + dr, pos[1] + dc

        if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
            pos = (nr, nc)
            path.append(pos)
        else:
            print("Invalid move")
            pygame.quit()
            return False

        clock.tick(60)

    pygame.quit()
    return reached_goal

