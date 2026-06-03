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

# Population for GA
def create_population(size, num_items):

    population = []
    base = list(range(num_items))
    
    for _ in range(size):
        individual = base[:]
        random.shuffle(individual)
        population.append(individual)
    
    return population

# Route distance for GA
def route_distance(route, distance_matrix):
    total = 0
    
    # start → first item
    total += distance_matrix[(0, route[0]+1)]
    
    # between items
    for i in range(len(route)-1):
        total += distance_matrix[(route[i]+1, route[i+1]+1)]
    
    # last item → back to start
    total += distance_matrix[(route[-1]+1, 0)]
    
    return total

# Parent Selection Types: Random, Tournament, Roulette, Rank, SUS
def random_sel(pop): return random.choice(pop)

def tournament_sel(pop, distance_matrix):

    return min(
        random.sample(pop, 3),
        key=lambda route:
            route_distance(route, distance_matrix)
    )

def roulette_sel(pop, distance_matrix):
    w = [
        1 / (route_distance(p, distance_matrix) + 1)
        for p in pop
    ]

    return random.choices(
        pop,
        weights=w
    )[0]

def rank_sel(pop, distance_matrix):

    s = sorted(
        pop,
        key=lambda route:
            route_distance(route, distance_matrix)
    )

    w = list(range(1, len(pop) + 1))

    return random.choices(
        s,
        weights=w
    )[0]

def sus_sel(pop, distance_matrix):

    w = [
        1 / (route_distance(p, distance_matrix) + 1)
        for p in pop
    ]

    total = sum(w)

    pick = random.uniform(0, total)

    cur = 0

    for i, p in enumerate(pop):

        cur += w[i]

        if cur > pick:
            return p

# Selecting the parent selection type to use for GA
def select(pop, selection_type, distance_matrix):

    if selection_type == "random":
        return random_sel(pop)

    if selection_type == "roulette":
        return roulette_sel(
            pop,
            distance_matrix
        )

    if selection_type == "sus":
        return sus_sel(
            pop,
            distance_matrix
        )

    if selection_type == "rank":
        return rank_sel(
            pop,
            distance_matrix
        )

    return tournament_sel(
        pop,
        distance_matrix
    )

# Crossover type for GA
def crossover(p1, p2, num_items):
    start, end = sorted(random.sample(range(num_items), 2))
    child = [None]*num_items
    
    child[start:end+1] = p1[start:end+1]
    
    fill = [x for x in p2 if x not in child]
    
    idx = 0
    for i in range(num_items):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    
    return child

# Mutation for GA
def mutate(route, mutation_rate):

    if random.random() < mutation_rate:

        i, j = random.sample(range(len(route)), 2)

        route[i], route[j] = route[j], route[i]

    return route


# GA with fitness
def genetic_algorithm_with_history(
    distance_matrix,
    items,
    generations=1000,
    size=500,
    mutation=0.1,
    selection_type="tournament"
):

    pop = create_population(size, len(items))

    history = []

    # Existing
    distance_history = []

    # New fitness statistics
    fit_best_history = []
    fit_avg_history = []
    fit_worst_history = []

    best = min(pop, key=lambda route: route_distance(route, distance_matrix))

    for g in range(generations):

        new_pop = []

        for _ in range(size):

            p1 = select(pop, selection_type, distance_matrix)
            p2 = select(pop, selection_type, distance_matrix)

            child = crossover(p1, p2, len(items))
            child = mutate(child, mutation)

            new_pop.append(child)

        pop = new_pop

        # Calculate fitness of current population
        population_distances = [
            route_distance(individual, distance_matrix)
            for individual in pop
        ]

        current = min(pop, key=lambda route: route_distance(route, distance_matrix))

        if route_distance(current, distance_matrix) < route_distance(best, distance_matrix):
            best = current

        # Existing history
        history.append(best.copy())
        distance_history.append(route_distance(best,distance_matrix))

        # New statistics
        fit_best_history.append(min(population_distances))
        fit_avg_history.append(
            sum(population_distances) / len(population_distances)
        )
        fit_worst_history.append(max(population_distances))

        if g % 10 == 0:
            print(f"Gen {g}: " f"{route_distance(best, distance_matrix)}")

    return (best, history, distance_history, fit_best_history, fit_avg_history, fit_worst_history)
    