from collections import deque
import random
import numpy as np

# BFS to find the furthest point from the start (1, 1)
# this Function was spat out from Chat
def bfs(start, rows, cols, maze):
    queue = deque([start])
    distances = np.full((rows, cols), -1)  # -1 means unvisited
    distances[start[0]][start[1]] = 0
    furthest_distance = 0
    furthest_point = start
    
    # Directions for 4-way movement (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        current_distance = distances[r][c]
        
        # Explore 4 directions
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0 and distances[nr][nc] == -1:
                distances[nr][nc] = current_distance + 1
                queue.append((nr, nc))
                
                # Check if this point is further than the previous furthest
                if distances[nr][nc] > furthest_distance:
                    furthest_distance = distances[nr][nc]
                    furthest_point = (nr, nc)
    
    return furthest_point

# baseline for visualizations, not actual mazes
def generate_random_maze(num_rows, num_cols):
    return np.random.randint(0, 2, size=(num_rows, num_cols))

# Returns the maze, start position, and goal position
def generate_maze_dfs_backtracker(num_rows, num_cols):
    # Ensure odd dimensions for a valid maze layout
    # this is because maze algorithms work best when dimensions are odd
    rows = num_rows if num_rows % 2 == 1 else num_rows + 1
    cols = num_cols if num_cols % 2 == 1 else num_cols + 1

    # Initialize a 2D array of 1s which are "walls". The open paths will be carved into this
    maze = [[1 for _ in range(cols)] for _ in range(rows)]
    

    # Helper to carve the open paths 
    def carve(r, c):
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]     # directions to carve. increments of 2 so we have space for walls between paths and not end up with a big block of empty space.
        random.shuffle(directions)      # pick a random direction to carve out 2 walls from
        for row_step, col_step in directions:
            end_row, end_col = r + row_step, c + col_step   # calculate the where we end up if we carve in this direction
            if 1 <= end_row < rows - 1 and 1 <= end_col < cols - 1 and maze[end_row][end_col] == 1:    # check if that cell is within bounds (excluding outermost walls) and if it is still a wall
                maze[end_row][end_col] = 0      # set the end cell to 0 (make it open)
                maze[r + row_step // 2][c + col_step // 2] = 0      # set the cells between where we started and where we ended to 0
                carve(end_row, end_col)

    # Start carving from (1,1)
    maze[1][1] = 0
    carve(1, 1)

    # Find the furthest point from the start using BFS
    goal = bfs((1, 1), rows, cols, maze)

    # Crop or pad back to desired size (Ideally not needed since params shuld have odd rows and cols.)
    return np.array(maze)[:num_rows, :num_cols], (1, 1), goal


def generate_prim_algo(num_rows, num_cols):
    maze = [[1 for _ in range(num_cols)] for _ in range(num_rows)]
    last_cell = (0, 0)

    cardinal_directions = [(2,0), (-2,0), (0,2), (0,-2)]

    walls = [(1,1,dx,dy) for dx,dy in cardinal_directions]     # add starting node to the list

    while walls:
        rand_idx = np.random.choice(len(walls))
        x,y,dx,dy = walls.pop(rand_idx)
        nx,ny = x+dx, y+dy
        print(f"x: {x}, y: {y}, nx: {nx}, ny:{ny}")

        if 0 <= nx < num_rows - 1 and 0 <= ny < num_cols - 1 and maze[nx][ny] == 1:
            maze[nx][ny] = 0    # make a new open cell
            last_cell = (nx,ny)
            maze[x+dx//2][y+dy//2] = 0      # carve path to the new opening
            for new_dx, new_dy in cardinal_directions:
                walls.append((nx, ny, new_dx, new_dy))

    return maze, (1,1), last_cell

