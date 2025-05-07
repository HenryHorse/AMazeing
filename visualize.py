import pygame
import time


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

