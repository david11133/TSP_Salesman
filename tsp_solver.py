import random
import math
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk 
import numpy as np
import threading
import time
from utils import calculate_distance, distance, draw_path, best_solution, get_best_solution, update_visuals

class TSPSolver:
    def __init__(self, update_callback=None):
        self.update_callback = update_callback
        self.population_size = 100
        self.generations = 500
        self.mutation_rate = 0.01
        self.best_distances = []
        self.cities = {}
        self.canvas = None
        self.root = None 
        self.current_best_tour = None

    def set_cities(self, cities):
        """Sets the cities from the GUI."""
        self.cities = cities
        if len(self.cities) < 2:
            messagebox.showerror("Insufficient Cities", "At least 2 cities are required to run the algorithm.")
            return

    def set_canvas(self, canvas, root=None):
        """Set the canvas for drawing paths."""
        self.canvas = canvas
        self.root = root

    def initialize_population(self, population_size):
        """Initialize the population for the genetic algorithm with unique paths."""
        cities_list = list(self.cities.keys())
        num_cities = len(cities_list)

        max_unique_permutations = math.factorial(num_cities)

        if population_size > max_unique_permutations:
            messagebox.showwarning("Adjusted Population Size", f"Maximum size for this poulation should be {max_unique_permutations}")
            return
        
        # Adjust population size if it's smaller than the number of cities
        if population_size < len(cities_list):
            messagebox.showwarning("Adjusted Population Size", f"Population size adjusted to {len(cities_list)} due to the number of cities.")
            population_size = len(cities_list)

        unique_population = set()

        # Generate the population until we have the desired number of unique paths
        while len(unique_population) < population_size:
            individual = tuple(random.sample(cities_list, len(cities_list)))            
            unique_population.add(individual)

        return list(unique_population)

    def evaluate_population(self, population):
        """Evaluate the fitness of each individual in the population."""
        fitness = []
        for tour in population:
            distance = calculate_distance(self.cities, tour)
            fitness.append(1 / distance)
        return fitness
    
    def selection(self, population, fitness):
        """Selects parents for crossover using tournament selection."""
        tournament_size = min(5, len(population))
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        tournament.sort(key=lambda x: x[1], reverse=True)
        return [tour for tour, _ in tournament[:2]]  # Select the best two parents

    def crossover(self, parents):
        """Apply Uniform Crossover ensuring unique offspring for a pair of parents."""
        parent1, parent2 = parents  # Expecting a pair of parents

        # Initialize the offspring chromosomes as empty lists
        child1 = [-1] * len(parent1)
        child2 = [-1] * len(parent2)

        # Create a random binary mask (alpha) to select genes from parents
        alpha = [random.choice([0, 1]) for _ in range(len(parent1))]

        # Fill in child1 and child2 based on the alpha mask
        for i in range(len(parent1)):
            if alpha[i] == 1:
                # Try to add from Parent 1 to Child 1, and Parent 2 to Child 2
                if parent1[i] not in child1:
                    child1[i] = parent1[i]
                else:
                    # If duplicate, add the next available gene from Parent 2
                    for gene in parent2:
                        if gene not in child1:
                            child1[i] = gene
                            break

                if parent2[i] not in child2:
                    child2[i] = parent2[i]
                else:
                    # If duplicate, add the next available gene from Parent 1
                    for gene in parent1:
                        if gene not in child2:
                            child2[i] = gene
                            break
            else:
                # Try to add from Parent 2 to Child 1, and Parent 1 to Child 2
                if parent2[i] not in child1:
                    child1[i] = parent2[i]
                else:
                    # If duplicate, add the next available gene from Parent 1
                    for gene in parent1:
                        if gene not in child1:
                            child1[i] = gene
                            break

                if parent1[i] not in child2:
                    child2[i] = parent1[i]
                else:
                    # If duplicate, add the next available gene from Parent 2
                    for gene in parent2:
                        if gene not in child2:
                            child2[i] = gene
                            break

        return [child1, child2]  # Return the two offspring

    def mutation(self, population):
        """Apply mutation (swap mutation) to the population."""
        for tour in population:
            if random.random() < self.mutation_rate:
                i, j = random.sample(range(len(tour)), 2)
                tour[i], tour[j] = tour[j], tour[i]
        return population

    def set_parameters(self, population_size, generations, mutation_rate):
        """Set the parameters for the genetic algorithm."""
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def run_algorithm(self, population_size=None, generations=None, mutation_rate=None):
        population_size = population_size or self.population_size
        generations = generations or self.generations
        mutation_rate = mutation_rate or self.mutation_rate

        population = self.initialize_population(population_size)
        self.best_distances = []
        self.current_best_tour = None

        for generation in range(generations):
            fitness = self.evaluate_population(population)
            parents = self.selection(population, fitness)
            
            next_generation = self.crossover(parents)
            next_generation = self.mutation(next_generation)

            next_generation[0] = best_solution(population, fitness)

            population = next_generation

            best_tour, best_distance = get_best_solution(self.cities, population)
            self.best_distances.append(best_distance)
            self.current_best_tour = best_tour

            update_visuals(self.canvas, self.root, self.update_callback, self.cities, best_tour, best_distance, self.current_best_tour, generation)


if __name__ == "__main__":
    app = TSPSolver()
