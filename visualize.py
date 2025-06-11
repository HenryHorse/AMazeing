import pygame
import time
import heapq
from models.helpers import manhattan_distance
from collections import deque




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


def run_mutating_visualization(initial_maze, rows, cols, mutator, start, goal, steps=20, interval_sec=3): 
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
            maze = mutator.mutate(maze, start, goal)  # unpack new_maze
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

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_maze(screen, maze, rows, cols)



        for r, c in path:
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (255, 0, 0), rect)

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

MOVES = [(-1,0),(1,0),(0,-1),(0,1)]

def astar_path(maze, start, goal):
    """
    A* search returning the full shortest path (list of (r,c) steps),
    or None if no path exists.
    """
    rows, cols = maze.shape
    open_set = [(manhattan_distance(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        if current in visited:
            continue
        visited.add(current)

        r, c = current
        for dr, dc in MOVES:
            nbr = (r + dr, c + dc)
            if (0 <= nbr[0] < rows and 0 <= nbr[1] < cols
                    and maze[nbr] == 0):
                tentative_g = g + 1
                if tentative_g < g_score.get(nbr, float('inf')):
                    g_score[nbr] = tentative_g
                    came_from[nbr] = current
                    f_score = tentative_g + manhattan_distance(nbr, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, nbr))

    return None



def Astar_and_mutator_visualization(mutator, maze, start, goal):
    """
    Live visualization where the SOLVER is plain A* and the MUTATOR applies
    a swap every 3 solver moves.
    """
    pygame.init()
    rows, cols = maze.shape
    screen_size = 600
    cell_size = screen_size // max(rows, cols)
    screen = pygame.display.set_mode((cols*cell_size, rows*cell_size))
    pygame.display.set_caption("A* Solver vs Mutator")

    clock = pygame.time.Clock()
    pos = start
    path = [pos]
    reached_goal = False
    move_counter = 0

    running = True
    while running:
        # handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # draw current maze
        draw_maze(screen, maze, rows, cols)

        # draw solver’s trail
        for (r, c) in path:
            rect = pygame.Rect(c*cell_size, r*cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (255,0,0), rect)

        # draw goal
        goal_color = (0,0,255) if reached_goal else (0,255,0)
        gr, gc = goal
        goal_rect = pygame.Rect(gc*cell_size, gr*cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, goal_color, goal_rect)

        pygame.display.flip()
        time.sleep(0.1)

        # if we’ve already reached the goal, just keep displaying
        if pos == goal:
            reached_goal = True
            continue

        # SOLVER MOVE: get full A* path, then step one cell
        full_path = astar_path(maze, pos, goal)
        if not full_path or len(full_path) < 2:
            print("No path found; terminating.")
            break

        pos = full_path[1]
        path.append(pos)
        move_counter += 1

        # MUTATOR: every 3 solver moves, apply one swap
        if move_counter % 3 == 0:
            maze = mutator.mutate(maze, pos, goal)

        clock.tick(60)

    pygame.quit()
    return reached_goal



def solver_and_mutator_visualization(solver, mutator, maze, start, goal):
    """
    Live visualization of a solver vs. mutator:
    - Solver moves one step per frame.
    - Mutator applies a mutation every 3 solver moves.
    """
    pygame.init()
    rows, cols = maze.shape
    screen_size = 600
    cell_size = screen_size // max(rows, cols)
    screen = pygame.display.set_mode((cols * cell_size, rows * cell_size))
    pygame.display.set_caption("Solver vs Mutator")

    clock = pygame.time.Clock()
    pos = start
    path = [pos]
    reached_goal = False
    move_counter = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw maze
        draw_maze(screen, maze, rows, cols)

        # Draw solver path
        for r, c in path:
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (255, 0, 0), rect)

        # Draw goal
        goal_color = (0, 0, 255) if reached_goal else (0, 255, 0)
        goal_rect = pygame.Rect(goal[1] * cell_size, goal[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, goal_color, goal_rect)

        pygame.display.flip()
        time.sleep(0.1)

        # Stop mutating once solver reaches goal
        if pos == goal:
            reached_goal = True
            continue

        # Solver step
        action = solver.get_next_move(maze, pos, start, goal)
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
            pos = (nr, nc)
            path.append(pos)
            move_counter += 1
        else:
            print("Invalid solver move; terminating visualization.")
            break

        # Mutate every 3 solver moves
        if move_counter % 3 == 0:
            maze = mutator.mutate(maze, pos, goal)
            # optionally print metrics:
            # print(f"Mutator applied at step {move_counter}: logp={logp.item():.3f}")

        clock.tick(60)

    pygame.quit()
    return reached_goal



def Astar_and_PPOmutator_visualization(mutator, maze, start, goal):
    """
    Live visualization where:
      - The solver is pure A* (one step per frame)
      - The mutator is a PPO agent that fires every 3 solver moves
      - Mutator.seen_swaps is reset at the start so swaps are unique per run
    """
    # 0) reset for unique‐swap constraint
    mutator.reset()

    pygame.init()
    rows, cols = maze.shape
    screen_size = 600
    cell_size   = screen_size // max(rows, cols)
    screen      = pygame.display.set_mode((cols*cell_size, rows*cell_size))
    pygame.display.set_caption("A* Solver vs PPO Mutator")

    clock       = pygame.time.Clock()
    pos         = start
    path        = [pos]
    reached_goal= False
    move_counter= 0

    running = True
    while running:
        # handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # draw maze background
        draw_maze(screen, maze, rows, cols)

        # draw solver’s trail
        for r, c in path:
            rect = pygame.Rect(c*cell_size, r*cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (255,0,0), rect)

        # draw goal cell
        goal_color = (0,0,255) if reached_goal else (0,255,0)
        gr, gc = goal
        goal_rect = pygame.Rect(gc*cell_size, gr*cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, goal_color, goal_rect)

        pygame.display.flip()
        time.sleep(0.1)

        # once solver reaches goal, just keep showing it
        if pos == goal:
            reached_goal = True
            continue

        # SOLVER MOVE: advance one step along A* path
        full_path = astar_path(maze, pos, goal)
        if not full_path or len(full_path) < 2:
            print("No path found; terminating.")
            break

        pos = full_path[1]
        path.append(pos)
        move_counter += 1

        # MUTATOR: every 3 solver moves, apply one PPO swap
        if move_counter % 3 == 0:
            # unpack properly so maze stays a numpy array
            new_maze, logp, value, entropy = mutator.mutate(maze, pos, goal)
            maze = new_maze
            # optional debugging:
            # print(f"Swap at step {move_counter}: logp={logp.item():.3f}, Δval={value.item():.3f}")

        clock.tick(60)

    pygame.quit()
    return reached_goal

