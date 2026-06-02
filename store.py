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
from astar import *

# Grid creation
grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]

# Function to add a shelf to the grid
def add_shelf(grid, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    for y in range(y1, y2):
        for x in range(x1, x2):
            grid[y][x] = 1

# Function where we create our layout with all our shelfs            
def create_store_layout(grid):
    add_shelf(grid, (2, 2), (4, 16))
    add_shelf(grid, (2, 18), (4, 20))
    add_shelf(grid, (6, 2), (8, 20))
    add_shelf(grid, (10, 2), (12, 6))
    add_shelf(grid, (10, 8), (12, 14))
    add_shelf(grid, (10, 16), (12, 20))
    add_shelf(grid, (14, 2), (16, 6))
    add_shelf(grid, (14, 8), (16, 14))
    add_shelf(grid, (14, 16), (16, 20))
    add_shelf(grid, (18, 2), (20, 6))
    add_shelf(grid, (18, 8), (20, 14))
    add_shelf(grid, (18, 16), (20, 20))
    add_shelf(grid, (22, 2), (34, 4))
    add_shelf(grid, (22, 6), (34, 8))
    add_shelf(grid, (22, 10), (34, 12))
    add_shelf(grid, (22, 14), (34, 16))
    add_shelf(grid, (22, 18), (34, 20))
    add_shelf(grid, (38, 2), (40, 6))
    add_shelf(grid, (38, 8), (40, 12))
    add_shelf(grid, (38, 14), (40, 18))
    add_shelf(grid, (38, 20), (40, 24)) 
    add_shelf(grid, (38, 28), (40, 30))
    add_shelf(grid, (38, 32), (40, 36))
    add_shelf(grid, (38, 38), (40, 40))
    add_shelf(grid, (2, 24), (4, 28))
    add_shelf(grid, (2, 30), (4, 32)) 
    add_shelf(grid, (2, 34), (4, 40))
    add_shelf(grid, (6, 24), (8, 28))
    add_shelf(grid, (6, 30), (8, 32)) 
    add_shelf(grid, (6, 34), (8, 40))
    add_shelf(grid, (10, 24), (12, 28))
    add_shelf(grid, (10, 30), (12, 32)) 
    add_shelf(grid, (10, 34), (12, 36))
    add_shelf(grid, (10, 38), (12, 40))
    add_shelf(grid, (14, 24), (16, 40))
    add_shelf(grid, (18, 24), (20, 26))
    add_shelf(grid, (18, 28), (20, 40))
    add_shelf(grid, (22, 24), (24, 26))
    add_shelf(grid, (22, 28), (24, 40))
    add_shelf(grid, (26, 24), (28, 26))
    add_shelf(grid, (26, 28), (28, 40))
    add_shelf(grid, (30, 24), (32, 26))
    add_shelf(grid, (30, 28), (32, 40))
    add_shelf(grid, (34, 24), (36, 26))
    add_shelf(grid, (34, 28), (36, 40))

# The agents start position
start = (41, 13)

# Each item location on the grid
items = [
    (13, 10),
    (40, 16),
    (4, 36),
    (13, 30),
    (37, 35),
    (24, 4),
    (5, 5),
    (8, 26),
    (40, 28),
    (24, 13),
    (9, 2),
    (37, 9),
    (30, 20),
    (8, 35),
    (20, 31),
    (20, 24),
    (33, 1),
    (20, 18),
    (32, 37),
    (16, 27)
    
]

# First 10 items
items_10 = items[:10]


random.seed(42)
# Item sets of random 5, 10, 15, and 20 items, for scalability analysis
item_sets = {
    5: random.sample(items, 5),
    10: random.sample(items, 10),
    15: random.sample(items, 15),
    20: items.copy()
}

def build_distance_matrix(grid, start, items):

    points = [start] + items

    distance_matrix = {}

    for i in range(len(points)):
        for j in range(len(points)):

            if i != j:

                _, distance = astar(
                    grid,
                    points[i],
                    points[j]
                )

                distance_matrix[(i, j)] = distance

    return distance_matrix