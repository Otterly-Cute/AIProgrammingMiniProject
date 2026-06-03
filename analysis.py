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
from ga import *

def benchmark_selection_methods(
    methods,
    distance_matrix,
    items,
    generations=100,
    size=100,
    runs=5,
    mutation=0.02
):

    results = {}
    convergence_data = {}

    for method in methods:

        distances = []
        runtimes = []
        all_histories = []

        print(f"\n===== Testing {method.upper()} =====")

        for r in range(runs):

            start_time = time.time()

            (
                best_route,
                history,
                distance_history,
                fit_best_history,
                fit_avg_history,
                fit_worst_history
            ) = genetic_algorithm_with_history(
                distance_matrix=distance_matrix,
                items=items,
                generations=generations,
                size=size,
                mutation=mutation,
                selection_type=method
            )

            runtime = time.time() - start_time

            best_distance = route_distance(
                best_route,
                distance_matrix
            )

            distances.append(best_distance)
            runtimes.append(runtime)

            all_histories.append(distance_history)

            print(
                f"Run {r+1}: "
                f"Distance = {best_distance:.1f}, "
                f"Time = {runtime:.2f}s"
            )

        results[method] = {
            "Average Distance": np.mean(distances),
            "Best Distance": np.min(distances),
            "Average Runtime (s)": np.mean(runtimes)
        }

        convergence_data[method] = np.mean(
            all_histories,
            axis=0
        )

    return results, convergence_data


methods = [
    "random",
    "tournament",
    "roulette",
    "rank",
    "sus"
]


def hyperparameter_analysis(
    distance_matrix,
    items,
    selection_type="tournament"
):

    population_sizes = [50, 100, 200, 500, 1000]
    mutation_rates = [0.01, 0.02, 0.05, 0.1, 0.2]
    generation_counts = [50, 100, 200, 500, 1000]

    results = []

    # -------------------------------------------------
    # Population Size Analysis
    # -------------------------------------------------

    print("\n===== POPULATION SIZE ANALYSIS =====")

    for pop_size in population_sizes:

        start_time = time.time()

        best_route, *_ = genetic_algorithm_with_history(
            distance_matrix=distance_matrix,
            items=items,
            generations=100,
            size=pop_size,
            mutation=0.02,
            selection_type=selection_type
        )

        runtime = time.time() - start_time

        distance = route_distance(
            best_route,
            distance_matrix
        )

        results.append({
            "Type": "Population Size",
            "Value": pop_size,
            "Distance": distance,
            "Runtime": runtime
        })

        print(
            f"Population {pop_size}: "
            f"Distance={distance}, "
            f"Time={runtime:.2f}s"
        )

    # -------------------------------------------------
    # Mutation Rate Analysis
    # -------------------------------------------------

    print("\n===== MUTATION RATE ANALYSIS =====")

    for rate in mutation_rates:

        start_time = time.time()

        best_route, *_ = genetic_algorithm_with_history(
            distance_matrix=distance_matrix,
            items=items,
            generations=100,
            size=100,
            mutation=rate,
            selection_type=selection_type
        )

        runtime = time.time() - start_time

        distance = route_distance(
            best_route,
            distance_matrix
        )

        results.append({
            "Type": "Mutation Rate",
            "Value": rate,
            "Distance": distance,
            "Runtime": runtime
        })

        print(
            f"Mutation {rate}: "
            f"Distance={distance}, "
            f"Time={runtime:.2f}s"
        )

    # -------------------------------------------------
    # Generation Count Analysis
    # -------------------------------------------------

    print("\n===== GENERATION COUNT ANALYSIS =====")

    for gens in generation_counts:

        start_time = time.time()

        best_route, *_ = genetic_algorithm_with_history(
            distance_matrix=distance_matrix,
            items=items,
            generations=gens,
            size=100,
            mutation=0.02,
            selection_type=selection_type
        )

        runtime = time.time() - start_time

        distance = route_distance(
            best_route,
            distance_matrix
        )

        results.append({
            "Type": "Generations",
            "Value": gens,
            "Distance": distance,
            "Runtime": runtime
        })

        print(
            f"Generations {gens}: "
            f"Distance={distance}, "
            f"Time={runtime:.2f}s"
        )

    return results

def parameter_grid_search(
    distance_matrix,
    items,
    selection_type="tournament"
):

    population_sizes = [50, 100, 200]
    mutation_rates = [0.01, 0.02, 0.05]
    generation_counts = [50, 100, 200]

    results = []

    for pop_size in population_sizes:
        for mut_rate in mutation_rates:
            for gens in generation_counts:

                start_time = time.time()

                best_route, *_ = genetic_algorithm_with_history(
                    distance_matrix=distance_matrix,
                    items=items,
                    generations=gens,
                    size=pop_size,
                    mutation=mut_rate,
                    selection_type=selection_type
                )

                runtime = time.time() - start_time

                distance = route_distance(
                    best_route,
                    distance_matrix
                )

                results.append({
                    "Population": pop_size,
                    "Mutation": mut_rate,
                    "Generations": gens,
                    "Distance": distance,
                    "Runtime": runtime
                })

                print(
                    f"Pop={pop_size}, "
                    f"Mut={mut_rate}, "
                    f"Gen={gens} "
                    f"-> Distance={distance:.1f}, "
                    f"Time={runtime:.2f}s"
                )

    return results

def find_best_hyperparameters(grid_results):

    best_result = min(
        grid_results,
        key=lambda x: x["Distance"]
    )

    optimized_result = min(
        grid_results,
        key=lambda x: x["Distance"] + 10 * x["Runtime"]
    )

    return best_result, optimized_result

def print_top_hyperparameter_combinations(
    grid_results,
    top_n=10
):

    sorted_results = sorted(
        grid_results,
        key=lambda x: x["Distance"]
    )

    print(f"\n===== TOP {top_n} COMBINATIONS =====")

    for rank, r in enumerate(sorted_results[:top_n], start=1):

        print(
            f"{rank:2d}. "
            f"Pop={r['Population']}, "
            f"Mut={r['Mutation']}, "
            f"Gen={r['Generations']} "
            f"-> Distance={r['Distance']:.1f}, "
            f"Runtime={r['Runtime']:.2f}s"
        )

    return sorted_results[:top_n]