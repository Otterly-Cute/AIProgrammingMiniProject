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
from astar import *

def build_full_path(
    route,
    grid,
    start,
    items
):

    full_path = []

    current = start

    for idx in route:

        target = items[idx]

        path_segment, _ = astar(
            grid,
            current,
            target
        )

        if not path_segment:
            print(f"No path from {current} to {target}")
            return []

        if full_path:
            path_segment = path_segment[1:]

        full_path.extend(path_segment)

        current = target

    # Return to start
    path_back, _ = astar(
        grid,
        current,
        start
    )

    if path_back:
        full_path.extend(path_back[1:])

    return full_path