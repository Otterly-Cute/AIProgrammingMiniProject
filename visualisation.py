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

# Function to visualize the grid with shelfs
def visualize_store(grid):
    height = len(grid)
    width = len(grid[0])
    
    # numeric grid
    heat = np.full((height, width), EMPTY, dtype=float)
    
    # shelves
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                heat[y][x] = WALL   # walls
    
    
    # plot
    plt.figure(figsize=(6,6))
    plt.imshow(heat, cmap = CMAP, vmin=-1, vmax=3)
    plt.title("Grid with Shelfs")
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Function to visualise the grid with shelfs, items, and start position
def visualize_items(grid, start, items):
    height = len(grid)
    width = len(grid[0])
    
    # numeric grid
    heat = np.full((height, width), EMPTY, dtype=float)
    
    # shelves
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                heat[y][x] = WALL   # walls
    
    # items
    for idx, (x, y) in enumerate(items):

        if 0 <= x < width and 0 <= y < height:

            heat[y][x] = ITEM
    
    # start
    sx, sy = start
    if EMPTY <= sx < width and EMPTY <= sy < height:
        heat[sy][sx] = START
    
    # plot
    plt.figure(figsize=(6,6))
    plt.imshow(heat, cmap=CMAP, vmin=-1, vmax=3)

    # label items with letters
    for idx, (x, y) in enumerate(items):

        letter = chr(ord('A') + idx)

        plt.text(
            x,
            y,
            letter,
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold"
        )
    
    plt.title("Grid with Items")
    plt.xticks([])
    plt.yticks([])
    plt.show()