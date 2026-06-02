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

# Function to visualize the distance matrix
def visualize_distance_matrix(distance_matrix, num_points):

    matrix = np.zeros((num_points, num_points))

    for (i, j), dist in distance_matrix.items():
        matrix[i][j] = dist

    plt.figure(figsize=(10,10))

    plt.imshow(matrix)

    plt.colorbar(label="A* Distance")

    for i in range(num_points):
        for j in range(num_points):

            if i != j:

                plt.text(
                    j,
                    i,
                    int(matrix[i][j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold"
                )

    plt.title("Distance Matrix")

    plt.xlabel("To")
    plt.ylabel("From")

    labels = ["SS"] + [
        chr(ord("A") + i)
        for i in range(num_points - 1)
    ]

    plt.xticks(range(num_points), labels)
    plt.yticks(range(num_points), labels)

    plt.show()