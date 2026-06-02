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
from store import *
from visualisation import *

# Creating the store layout with shelfes
create_store_layout(grid)

# visualizing the store with shelfes
visualize_store(grid)

# Visualizing the items placement in the store (first 10)
visualize_items(grid, start, items_10)

# Visualizing the items placement in the store (all 20)
visualize_items(grid, start, items)

# Creates a distance matrix of the items with A* (first 10)
distance_matrix_10 = build_distance_matrix(grid, start, items_10)

# Creates a distance matrix of the items with A* (all 20)
distance_matrix_all = build_distance_matrix(grid, start, items)