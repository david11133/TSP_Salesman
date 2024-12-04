import random
import math
import tkinter as tk
from tkinter import messagebox
from utils import calculate_distance, best_solution, get_best_solution, update_visuals


class Population:
    def __init__(self, cities, size):
        self.cities = cities
        self.size = size
        self.individuals = self.generate_population()

    def generate_population(self):
        """Generate a population of unique individuals."""
        city_list = list(self.cities.keys())
        max_permutations = math.factorial(len(city_list))
        
        if self.size > max_permutations:
            messagebox.showwarning("Population Size", f"Maximum allowed population size is {max_permutations}")
            return []

        if self.size < len(city_list):
            messagebox.showwarning("Population Size", "Population size adjusted to the number of cities.")
            self.size = len(city_list)

        population = set()
        while len(population) < self.size:
            individual = tuple(random.sample(city_list, len(city_list)))
            population.add(individual)

        return list(population)

    def evaluate(self):
        """Calculate fitness of each individual."""
        fitness_scores = []
        for individual in self.individuals:
            distance = calculate_distance(self.cities, individual)
            fitness_scores.append(1 / distance)
        return fitness_scores

    def select_parents(self, fitness_scores):
        """Tournament selection to choose parents."""
        tournament_size = min(5, len(self.individuals))
        tournament = random.sample(list(zip(self.individuals, fitness_scores)), tournament_size)
        tournament.sort(key=lambda x: x[1], reverse=True)
        return [ind for ind, _ in tournament[:2]]


class GeneticAlgorithm:
    def __init__(self, population, mutation_rate=0.01, update_callback=None):
        self.population = population
        self.mutation_rate = mutation_rate
        self.update_callback = update_callback
        self.best_tour = None
        self.best_distance = float('inf')
        self.best_distances = []

    def crossover(self, parents):
        """Perform uniform crossover."""
        parent1, parent2 = parents
        child1, child2 = [-1] * len(parent1), [-1] * len(parent2)
        alpha = [random.choice([0, 1]) for _ in range(len(parent1))]

        for i in range(len(parent1)):
            if alpha[i] == 1:
                child1[i] = parent1[i] if parent1[i] not in child1 else next(gene for gene in parent2 if gene not in child1)
                child2[i] = parent2[i] if parent2[i] not in child2 else next(gene for gene in parent1 if gene not in child2)
            else:
                child1[i] = parent2[i] if parent2[i] not in child1 else next(gene for gene in parent1 if gene not in child1)
                child2[i] = parent1[i] if parent1[i] not in child2 else next(gene for gene in parent2 if gene not in child2)

        return [child1, child2]

    def mutate(self, population):
        """Apply mutation to the population."""
        for individual in population:
            if random.random() < self.mutation_rate:
                i, j = random.sample(range(len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]
        return population

    def run_generation(self):
        """Run one generation of the genetic algorithm."""
        fitness_scores = self.population.evaluate()
        parents = self.population.select_parents(fitness_scores)
        next_generation = self.crossover(parents)
        next_generation = self.mutate(next_generation)

        # Update the population with the best individual
        best_individual = best_solution(self.population.individuals, fitness_scores)
        next_generation[0] = best_individual

        self.population.individuals = next_generation
        return fitness_scores


class TSPSolver:
    def __init__(self, update_callback=None):
        self.update_callback = update_callback
        self.cities = {}
        self.canvas = None
        self.root = None
        self.genetic_algorithm = None

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

    def run_algorithm(self, population_size=100, generations=500, mutation_rate=0.01):
        """Run the genetic algorithm."""
        population = Population(self.cities, population_size)
        self.genetic_algorithm = GeneticAlgorithm(population, mutation_rate, self.update_callback)
        self.best_distances = []

        for generation in range(generations):
            fitness_scores = self.genetic_algorithm.run_generation()
            best_tour, best_distance = get_best_solution(self.cities, population.individuals)

            self.best_distances.append(best_distance)
            self.genetic_algorithm.best_tour = best_tour
            self.genetic_algorithm.best_distance = best_distance

            update_visuals(self.canvas, self.root, self.update_callback, self.cities, best_tour, best_distance)


if __name__ == "__main__":
    app = TSPSolver()
