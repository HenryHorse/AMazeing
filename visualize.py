import pygame



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