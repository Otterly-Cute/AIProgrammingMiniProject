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

# Using these same values everywhere, to keep consistency between plots and animations
EMPTY = 0
WALL = -1
ITEM = 2
ITEM_PICKED = 1.2
START = 2.2
AGENT = 3

CMAP = "GnBu"


# Grid Settings
grid_width = 42
grid_height = 42