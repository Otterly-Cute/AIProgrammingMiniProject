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
def create_population(size):
    population = []
    base = list(range(NUM_ITEMS))
    
    for _ in range(size):
        individual = base[:]
        random.shuffle(individual)
        population.append(individual)
    
    return population

# Route distance for GA
def route_distance(route):
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

def tournament_sel(pop):
    return min(random.sample(pop,3), key=route_distance)

def roulette_sel(pop):
    w=[1/(route_distance(p)+1) for p in pop]
    return random.choices(pop,weights=w)[0]

def rank_sel(pop):
    s=sorted(pop,key=route_distance)
    w=list(range(1,len(pop)+1))
    return random.choices(s,weights=w)[0]

def sus_sel(pop):
    w=[1/(route_distance(p)+1) for p in pop]
    total=sum(w)
    pick=random.uniform(0,total)
    cur=0
    for i,p in enumerate(pop):
        cur+=w[i]
        if cur>pick: return p

# Selecting the parent selection type to use for GA
def select(pop):
    if SELECTION=="random": return random_sel(pop)
    if SELECTION=="roulette": return roulette_sel(pop)
    if SELECTION=="sus": return sus_sel(pop)
    if SELECTION=="rank": return rank_sel(pop)
    return tournament_sel(pop)

# Crossover type for GA
def crossover(p1, p2):
    start, end = sorted(random.sample(range(NUM_ITEMS), 2))
    child = [None]*NUM_ITEMS
    
    child[start:end+1] = p1[start:end+1]
    
    fill = [x for x in p2 if x not in child]
    
    idx = 0
    for i in range(NUM_ITEMS):
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
    generations=1000,
    size=500,
    mutation=0.1
):

    pop = create_population(size)

    history = []

    # Existing
    distance_history = []

    # New fitness statistics
    fit_best_history = []
    fit_avg_history = []
    fit_worst_history = []

    best = min(pop, key=route_distance)

    for g in range(generations):

        new_pop = []

        for _ in range(size):

            p1 = select(pop)
            p2 = select(pop)

            child = crossover(p1, p2)
            child = mutate(child, mutation)

            new_pop.append(child)

        pop = new_pop

        # Calculate fitness of current population
        population_distances = [
            route_distance(individual)
            for individual in pop
        ]

        current = min(pop, key=route_distance)

        if route_distance(current) < route_distance(best):
            best = current

        # Existing history
        history.append(best.copy())
        distance_history.append(route_distance(best))

        # New statistics
        fit_best_history.append(min(population_distances))
        fit_avg_history.append(
            sum(population_distances) / len(population_distances)
        )
        fit_worst_history.append(max(population_distances))

        if g % 10 == 0:
            print(f"Gen {g}: {route_distance(best)}")

    return (
        best,
        history,
        distance_history,
        fit_best_history,
        fit_avg_history,
        fit_worst_history
    )