# Packages
import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
import matplotlib.animation as animation
from IPython.display import HTML

# Other scripts
from config import *

# ????
def get_neighbors(pos):
    x, y = pos
    
    candidates = [
        (x+1, y),
        (x-1, y),
        (x, y+1),
        (x, y-1)
    ]
    
    valid = []
    for nx, ny in candidates:
        if 0 <= nx < grid_width and 0 <= ny < grid_height:
            valid.append((nx, ny))
    
    return valid

# ????
def heuristic(a, b):
    ax, ay = a
    bx, by = b
    return abs(ax - bx) + abs(ay - by)

# A* funtion
def astar(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # reconstruct path
            path = []
            temp = current
            
            while temp in came_from:
                path.append(temp)
                temp = came_from[temp]
            
            path.append(start)
            path.reverse()
            
            distance = g_score[current]
            
            return path, distance
        
        for neighbor in get_neighbors(current):
            x, y = neighbor
            
            # bounds
            if not (0 <= x < len(grid[0]) and 0 <= y < len(grid)):
                continue
            
            # wall
            if grid[y][x] == 1:
                continue
            
            tentative = g_score[current] + 1
            
            if neighbor not in g_score or tentative < g_score[neighbor]:
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                
                f = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
    
    return [], float("inf")