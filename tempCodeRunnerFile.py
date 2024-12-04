def run_genetic_algorithm(self):
        """Start the genetic algorithm with the user's input."""
        population_size = int(self.pop_size_entry.get())
        generations = int(self.gen_entry.get())
        mutation_rate = float(self.mutation_rate_entry.get())

        if not self._validate_inputs(population_size, generations, mutation_rate):
            return

        self.solver.set_cities(self.cities)
        self.solver.set_canvas(self.canvas)
        self.solver.run_algorithm(population_size, generations + 1, mutation_rate)