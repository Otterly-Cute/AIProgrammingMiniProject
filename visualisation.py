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
from path import *
from ga import *

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

def plot_selection_comparison(results):

    methods = list(results.keys())

    avg_distances = [
        results[m]["Average Distance"]
        for m in methods
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(methods, avg_distances)

    plt.ylabel("Average Distance")
    plt.xlabel("Selection Method")

    plt.title("Selection Method Distance Comparison")

    plt.grid(axis="y")

    plt.show()

def plot_selection_runtime(results):

    methods = list(results.keys())

    runtimes = [
        results[m]["Average Runtime (s)"]
        for m in methods
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(methods, runtimes)

    plt.ylabel("Average Runtime (s)")
    plt.xlabel("Selection Method")

    plt.title("Selection Method Runtime Comparison")

    plt.grid(axis="y")

    plt.show()

def plot_distance_runtime_tradeoff(results):

    plt.figure(figsize=(8, 6))

    for method in results.keys():

        distance = results[method]["Average Distance"]
        runtime = results[method]["Average Runtime (s)"]

        plt.scatter(
            runtime,
            distance,
            s=120
        )

        plt.annotate(
            method.capitalize(),
            (runtime, distance),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Average Runtime (s)")
    plt.ylabel("Average Distance")

    plt.title("Distance vs Runtime Comparison")

    plt.grid(True)

    plt.show()

def plot_population_size_analysis(results):

    pop_results = [
        r for r in results
        if r["Type"] == "Population Size"
    ]

    x = [r["Value"] for r in pop_results]
    y = [r["Distance"] for r in pop_results]

    plt.figure(figsize=(8, 5))

    plt.plot(x, y, marker="o")

    plt.xlabel("Population Size")
    plt.ylabel("Distance")

    plt.title("Population Size Analysis")

    plt.grid(True)

    plt.show()

def plot_mutation_rate_analysis(results):

    mutation_results = [
        r for r in results
        if r["Type"] == "Mutation Rate"
    ]

    x = [r["Value"] for r in mutation_results]
    y = [r["Distance"] for r in mutation_results]

    plt.figure(figsize=(8, 5))

    plt.plot(x, y, marker="o")

    plt.xlabel("Mutation Rate")
    plt.ylabel("Distance")

    plt.title("Mutation Rate Analysis")

    plt.grid(True)

    plt.show()

def plot_generation_analysis(results):

    generation_results = [
        r for r in results
        if r["Type"] == "Generations"
    ]

    x = [r["Value"] for r in generation_results]
    y = [r["Distance"] for r in generation_results]

    plt.figure(figsize=(8, 5))

    plt.plot(x, y, marker="o")

    plt.xlabel("Number of Generations")
    plt.ylabel("Distance")

    plt.title("Generation Count Analysis")

    plt.grid(True)

    plt.show()

def plot_population_runtime(results):

    pop_results = [
        r for r in results
        if r["Type"] == "Population Size"
    ]

    x = [r["Value"] for r in pop_results]
    y = [r["Runtime"] for r in pop_results]

    plt.figure(figsize=(8, 5))

    plt.plot(x, y, marker="o")

    plt.xlabel("Population Size")
    plt.ylabel("Runtime (s)")

    plt.title("Population Size vs Runtime")

    plt.grid(True)

    plt.show()

def plot_mutation_runtime(results):

    mutation_results = [
        r for r in results
        if r["Type"] == "Mutation Rate"
    ]

    x = [r["Value"] for r in mutation_results]
    y = [r["Runtime"] for r in mutation_results]

    plt.figure(figsize=(8, 5))

    plt.plot(x, y, marker="o")

    plt.xlabel("Mutation Rate")
    plt.ylabel("Runtime (s)")

    plt.title("Mutation Rate vs Runtime")

    plt.grid(True)

    plt.show()

def plot_generation_runtime(results):

    generation_results = [
        r for r in results
        if r["Type"] == "Generations"
    ]

    x = [r["Value"] for r in generation_results]
    y = [r["Runtime"] for r in generation_results]

    plt.figure(figsize=(8, 5))

    plt.plot(x, y, marker="o")

    plt.xlabel("Number of Generations")
    plt.ylabel("Runtime (s)")

    plt.title("Generation Count vs Runtime")

    plt.grid(True)

    plt.show()

def plot_population_mutation_heatmap(
    grid_results,
    generations=100
):

    population_sizes = sorted(
        list(set(r["Population"] for r in grid_results))
    )

    mutation_rates = sorted(
        list(set(r["Mutation"] for r in grid_results))
    )

    heat = np.zeros(
        (len(population_sizes),
         len(mutation_rates))
    )

    for i, pop in enumerate(population_sizes):
        for j, mut in enumerate(mutation_rates):

            matching = [
                r for r in grid_results
                if r["Population"] == pop
                and r["Mutation"] == mut
                and r["Generations"] == generations
            ]

            if matching:
                heat[i][j] = matching[0]["Distance"]

    plt.figure(figsize=(8, 6))

    plt.imshow(heat)

    plt.colorbar(label="Distance")

    plt.xticks(
        range(len(mutation_rates)),
        mutation_rates
    )

    plt.yticks(
        range(len(population_sizes)),
        population_sizes
    )

    plt.xlabel("Mutation Rate")
    plt.ylabel("Population Size")

    plt.title(
        f"Distance Heatmap (Generations = {generations})"
    )

    for i in range(len(population_sizes)):
        for j in range(len(mutation_rates)):

            plt.text(
                j,
                i,
                f"{heat[i,j]:.0f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold"
            )

    plt.show()

def plot_population_generation_heatmap(
    grid_results,
    mutation=0.02
):

    population_sizes = sorted(
        list(set(r["Population"] for r in grid_results))
    )

    generation_counts = sorted(
        list(set(r["Generations"] for r in grid_results))
    )

    heat = np.zeros(
        (len(population_sizes),
         len(generation_counts))
    )

    for i, pop in enumerate(population_sizes):
        for j, gen in enumerate(generation_counts):

            matching = [
                r for r in grid_results
                if r["Population"] == pop
                and r["Generations"] == gen
                and r["Mutation"] == mutation
            ]

            if matching:
                heat[i][j] = matching[0]["Distance"]

    plt.figure(figsize=(8, 6))

    plt.imshow(heat)

    plt.colorbar(label="Distance")

    plt.xticks(
        range(len(generation_counts)),
        generation_counts
    )

    plt.yticks(
        range(len(population_sizes)),
        population_sizes
    )

    plt.xlabel("Generations")
    plt.ylabel("Population Size")

    plt.title(
        f"Distance Heatmap (Mutation = {mutation})"
    )

    for i in range(len(population_sizes)):
        for j in range(len(generation_counts)):

            plt.text(
                j,
                i,
                f"{heat[i,j]:.0f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold"
            )

    plt.show()

def plot_mutation_generation_heatmap(
    grid_results,
    population=100
):

    mutation_rates = sorted(
        list(set(r["Mutation"] for r in grid_results))
    )

    generation_counts = sorted(
        list(set(r["Generations"] for r in grid_results))
    )

    heat = np.zeros(
        (len(mutation_rates),
         len(generation_counts))
    )

    for i, mut in enumerate(mutation_rates):
        for j, gen in enumerate(generation_counts):

            matching = [
                r for r in grid_results
                if r["Mutation"] == mut
                and r["Generations"] == gen
                and r["Population"] == population
            ]

            if matching:
                heat[i][j] = matching[0]["Distance"]

    plt.figure(figsize=(8, 6))

    plt.imshow(heat)

    plt.colorbar(label="Distance")

    plt.xticks(
        range(len(generation_counts)),
        generation_counts
    )

    plt.yticks(
        range(len(mutation_rates)),
        mutation_rates
    )

    plt.xlabel("Generations")
    plt.ylabel("Mutation Rate")

    plt.title(
        f"Distance Heatmap (Population = {population})"
    )

    for i in range(len(mutation_rates)):
        for j in range(len(generation_counts)):

            plt.text(
                j,
                i,
                f"{heat[i,j]:.0f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold"
            )

    plt.show()

def plot_runtime_distance_scatter(
    grid_results,
    top_n=10
):

    plt.figure(figsize=(10, 8))

    # Plot all results
    plt.scatter(
        [r["Runtime"] for r in grid_results],
        [r["Distance"] for r in grid_results]
    )

    # Sort by distance
    best_results = sorted(
        grid_results,
        key=lambda x: x["Distance"]
    )[:top_n]

    # Best overall configuration
    best_result = best_results[0]

    # Highlight best point
    plt.scatter(
        best_result["Runtime"],
        best_result["Distance"],
        s=250
    )

    # Number the top results
    for rank, r in enumerate(best_results, start=1):

        plt.annotate(
            str(rank),
            (r["Runtime"], r["Distance"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold"
        )

    plt.xlabel("Runtime (s)")
    plt.ylabel("Distance")

    plt.title(
        "Runtime vs Distance for Hyperparameter Combinations"
    )

    plt.grid(True)

    plt.show()

    # Print legend underneath
    print(f"\n===== TOP {top_n} CONFIGURATIONS =====")

    for rank, r in enumerate(best_results, start=1):

        print(
            f"{rank:2d}. "
            f"Pop={r['Population']}, "
            f"Mut={r['Mutation']}, "
            f"Gen={r['Generations']} "
            f"-> Distance={r['Distance']:.1f}, "
            f"Runtime={r['Runtime']:.2f}s"
        )

    return best_results

def visualize_path(
    grid,
    full_path,
    start,
    items,
    route,
    title="Picking Route"
):
    plt.close('all')

    height = len(grid)
    width = len(grid[0])

    heat = np.full(
        (height, width),
        EMPTY,
        dtype=float
    )

    # shelves
    for y in range(height):
        for x in range(width):

            if grid[y][x] == 1:
                heat[y][x] = WALL

    # items
    for (x, y) in items:

        if 0 <= x < width and 0 <= y < height:
            heat[y][x] = ITEM

    # start
    sx, sy = start

    if 0 <= sx < width and 0 <= sy < height:
        heat[sy][sx] = START

    # path
    for (x, y) in full_path:

        if 0 <= x < width and 0 <= y < height:

            if (x, y) != start and (x, y) not in items:
                heat[y][x] = AGENT

    # ---------- PLOT ----------
    fig, ax = plt.subplots(figsize=(8,8))

    ax.imshow(
        heat,
        cmap=CMAP,
        vmin=-1,
        vmax=3
    )

    # item letter + visit order
    for order, item_idx in enumerate(route):

        x, y = items[item_idx]

        label = f"{chr(65 + item_idx)}\n{order + 1}"

        ax.text(
            x,
            y,
            label,
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold"
        )

    ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    fig.show()

def animate_heatmap_funcanimation(
    grid,
    path,
    start,
    items,
    trail_length=10,
    interval=50
):

    if not path:
        print("Path is empty.")
        return None

    height = len(grid)
    width = len(grid[0])

    item_set = set(items)

    # ---------- STATIC BASE ----------
    base_heat = np.full(
        (height, width),
        EMPTY,
        dtype=float
    )

    for y in range(height):
        for x in range(width):

            if grid[y][x] == 1:
                base_heat[y][x] = WALL

    # ---------- FIGURE ----------
    fig, ax = plt.subplots(figsize=(6, 6))

    heat = base_heat.copy()

    img = ax.imshow(
        heat,
        cmap=CMAP,
        vmin=-1,
        vmax=3,
        animated=True
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # ---------- ITEM LABELS ----------
    for idx, (x, y) in enumerate(items):

        ax.text(
            x,
            y,
            chr(65 + idx),
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold"
        )

    # ---------- UPDATE FUNCTION ----------
    def update(frame):

        current = path[frame]

        heat = base_heat.copy()

        # ---------- TRAIL ----------
        start_idx = max(0, frame - trail_length)

        trail = path[start_idx:frame + 1]

        for i, (x, y) in enumerate(trail):

            intensity = (i + 1) / len(trail)

            heat[y][x] = 0.2 + intensity * 1.8

        # ---------- VISITED ITEMS ----------
        visited_items = set(path[:frame + 1]) & item_set

        for (x, y) in items:

            if (x, y) in visited_items:
                heat[y][x] = ITEM_PICKED
            else:
                heat[y][x] = ITEM

        # ---------- START ----------
        sx, sy = start
        heat[sy][sx] = START

        # ---------- AGENT ----------
        x, y = current
        heat[y][x] = AGENT

        # ---------- UPDATE IMAGE ----------
        img.set_array(heat)

        ax.set_title(
            f"Step {frame + 1}/{len(path)} | "
            f"Collected {len(visited_items)}/{len(items)} Items"
        )

        return [img]

    # ---------- CREATE ANIMATION ----------
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(path),
        interval=interval,
        blit=False,
        repeat=False
    )

    return ani

def animate_ga_evolution(
    grid,
    history,
    distance_history,
    start,
    items,
    distance_matrix,
    interval=150,
    generation_step=5
):

    height = len(grid)
    width = len(grid[0])

    # ---------- GENERATIONS TO SHOW ----------
    shown_generations = list(
        range(0, len(history), generation_step)
    )

    # ---------- STATIC BASE ----------
    base_heat = np.full(
        (height, width),
        EMPTY,
        dtype=float
    )

    for y in range(height):
        for x in range(width):

            if grid[y][x] == 1:
                base_heat[y][x] = WALL

    # ---------- FIGURE ----------
    fig = plt.figure(figsize=(12, 6))

    ax_grid = fig.add_subplot(1, 2, 1)
    ax_graph = fig.add_subplot(1, 2, 2)

    heat = base_heat.copy()

    img = ax_grid.imshow(
        heat,
        cmap=CMAP,
        vmin=-1,
        vmax=3,
        animated=True
    )

    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    # ---------- GRAPH ----------
    ax_graph.set_title("GA Convergence")

    ax_graph.set_xlabel("Generation")
    ax_graph.set_ylabel("Distance")

    ax_graph.set_xlim(0, len(history))

    ax_graph.set_ylim(
        min(distance_history) * 0.95,
        max(distance_history) * 1.05
    )

    line, = ax_graph.plot([], [])

    # ---------- UPDATE ----------
    def update(frame_idx):

        gen = shown_generations[frame_idx]

        route = history[gen]

        heat = base_heat.copy()

        # ---------- BUILD PATH ----------
        full_path = build_full_path(
            route,
            grid,
            start,
            items
        )

        generation_progress = gen / len(history)

        path_value = 0.5 + generation_progress * 2.0

        # ---------- DRAW PATH ----------
        for (x, y) in full_path:

            if (x, y) != start and (x, y) not in items:
                heat[y][x] = path_value

        # ---------- ITEMS ----------
        for item in items:

            x, y = item

            heat[y][x] = ITEM

        # ---------- START ----------
        sx, sy = start

        heat[sy][sx] = START

        # ---------- UPDATE GRID ----------
        img.set_array(heat)

        # ---------- REMOVE OLD LABELS ----------
        for txt in ax_grid.texts:
            txt.remove()

        # ---------- ITEM LETTERS + VISIT ORDER ----------
        for order, item_idx in enumerate(route):

            x, y = items[item_idx]

            label = (
                f"{chr(65 + item_idx)}\n"
                f"{order + 1}"
            )

            ax_grid.text(
                x,
                y,
                label,
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold"
            )

        # ---------- GRID TITLE ----------
        ax_grid.set_title(
            f"Generation {gen + 1}/{len(history)}\n"
            f"Distance: "
            f"{route_distance(route, distance_matrix):.0f}"
        )

        # ---------- UPDATE GRAPH ----------
        x_data = list(range(gen + 1))

        y_data = distance_history[:gen + 1]

        line.set_data(
            x_data,
            y_data
        )

        return [img, line]

    # ---------- CREATE ANIMATION ----------
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(shown_generations),
        interval=interval,
        blit=False,
        repeat=False
    )

    return ani

def plot_scalability_summary(scale_results):

    items = [r["Items"] for r in scale_results]
    runtimes = [r["Runtime"] for r in scale_results]
    distances = [r["Distance"] for r in scale_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Runtime (left axis)
    line1 = ax1.plot(
        items,
        runtimes,
        marker="o",
        color="tab:blue",
        label="Runtime (s)"
    )

    ax1.set_xlabel("Number of Items")
    ax1.set_ylabel(
        "Runtime (s)",
        color="tab:blue"
    )

    ax1.tick_params(
        axis="y",
        labelcolor="tab:blue"
    )

    # Distance (right axis)
    ax2 = ax1.twinx()

    line2 = ax2.plot(
        items,
        distances,
        marker="s",
        color="tab:red",
        label="Distance"
    )

    ax2.set_ylabel(
        "Route Distance",
        color="tab:red"
    )

    ax2.tick_params(
        axis="y",
        labelcolor="tab:red"
    )

    # Combined legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]

    ax1.legend(
        lines,
        labels,
        loc="upper left"
    )

    plt.title("GA Scalability Analysis")

    plt.grid(True)

    plt.show()