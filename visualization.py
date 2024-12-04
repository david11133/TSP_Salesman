import tkinter as tk
from tkinter import messagebox, ttk
import random
import string
from tsp_solver import TSPSolver

# Default values for the GUI
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BTN_WIDTH = 30
BTN_HEIGHT = 1

class TSPGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Traveling Salesman Problem with Genetic Algorithm")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        self.cities = {}
        self.solver = TSPSolver(update_callback=self.update_stats)

        self.create_widgets()

    def create_widgets(self):
        """Creates and arranges the main UI components."""
        self.control_frame = tk.Frame(self, bg="#34495E")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.canvas_frame, bg="#ECF0F1", scrollregion=(0, 0, 1000, 1000)
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        self.canvas_scrollbar = ttk.Scrollbar(
            self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas_scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.config(yscrollcommand=self.canvas_scrollbar.set)

        self.canvas_hscrollbar = ttk.Scrollbar(
            self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        self.canvas_hscrollbar.grid(row=1, column=0, sticky="ew")
        self.canvas.config(xscrollcommand=self.canvas_hscrollbar.set)

        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.create_stats_labels()
        self.create_input_fields()
        self.create_buttons()

    def create_stats_labels(self):
        """Creates labels to display statistics."""
        self.stats_frame = tk.Frame(self.control_frame, bg="#34495E")
        self.stats_frame.pack(fill=tk.Y, padx=5, pady=10)

        # Best Tour Label with bigger font and bold
        self.best_tour_label = tk.Label(self.stats_frame, text="Best Tour: ", anchor="w", wraplength=150, bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        self.best_tour_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # Distance Label with bigger font and bold
        self.distance_label = tk.Label(self.stats_frame, text="Distance: ", anchor="w", bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        self.distance_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)

    def create_input_fields(self):
        """Create input fields for population size, number of generations, and mutation rate."""
        input_frame = tk.Frame(self.control_frame, bg="#34495E")
        input_frame.pack(fill=tk.Y, padx=5, pady=(80, 10))

        # Population Size
        self.pop_size_label = tk.Label(input_frame, text="Population Size:", bg="#34495E", fg="white", font=("Arial", 10))
        self.pop_size_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.pop_size_entry = tk.Entry(input_frame, bg="#ffffff", fg="black")
        self.pop_size_entry.grid(row=0, column=1, padx=5, pady=5)
        self.pop_size_entry.insert(0, "50")  # Default value

        # Number of Generations
        self.gen_label = tk.Label(input_frame, text="Number of Generations:", bg="#34495E", fg="white", font=("Arial", 10))
        self.gen_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.gen_entry = tk.Entry(input_frame, bg="#ffffff", fg="black")
        self.gen_entry.grid(row=1, column=1, padx=5, pady=5)
        self.gen_entry.insert(0, "20")

        # Mutation Rate
        self.mutation_rate_label = tk.Label(input_frame, text="Mutation Rate:", bg="#34495E", fg="white", font=("Arial", 10))
        self.mutation_rate_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.mutation_rate_entry = tk.Entry(input_frame, bg="#ffffff", fg="black")
        self.mutation_rate_entry.grid(row=2, column=1, padx=5, pady=5)
        self.mutation_rate_entry.insert(0, "0.01")

    def create_buttons(self):
        """Create action buttons for controlling the simulation."""
        self.buttons_frame = tk.Frame(self.control_frame, bg="#34495E")
        self.buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Start the Algorithm
        self.start_btn = tk.Button(
            self.buttons_frame,
            text="Start",
            command=self.run_genetic_algorithm,
            width=BTN_WIDTH,
            height=BTN_HEIGHT,
            bg="#1ABC9C",
            fg="white"
        )
        self.start_btn.pack(pady=10)

        # Add City
        self.add_city_btn = tk.Button(
            self.buttons_frame,
            text="Add City",
            command=self.add_random_city,
            width=BTN_WIDTH,
            height=BTN_HEIGHT,
            bg="#E67E22",
            fg="white"
        )
        self.add_city_btn.pack(pady=10)

        # Remove City
        self.remove_city_btn = tk.Button(
            self.buttons_frame,
            text="Remove City",
            command=self.remove_city,
            width=BTN_WIDTH,
            height=BTN_HEIGHT,
            bg="#E74C3C",
            fg="white"
        )
        self.remove_city_btn.pack(pady=10)

    def run_genetic_algorithm(self):
        """Starts the genetic algorithm."""
        # Retrieve values from the entry fields
        population_size = int(self.pop_size_entry.get())
        generations = int(self.gen_entry.get())
        mutation_rate = float(self.mutation_rate_entry.get())

        # Validate inputs before running the algorithm
        if population_size <= 0 or generations <= 0 or not (0 <= mutation_rate <= 1):
            messagebox.showerror("Invalid Input", "Please enter valid values for population size, generations, and mutation rate.")
            return

        # Ensure population size is at least as large as the number of cities
        if population_size < len(self.cities):
            messagebox.showerror("Invalid Population Size", "Population size must be greater than or equal to the number of cities.")
            return

        # Pass the cities and canvas to the solver
        self.solver.set_cities(self.cities)
        self.solver.set_canvas(self.canvas)

        # Call the run_algorithm method with the values entered by the user
        self.solver.run_algorithm(population_size, generations + 1, mutation_rate)

    def update_stats(self, best_tour, current_distance, generation):
        """Update the GUI with the best tour and its distance."""
        self.best_tour_label.config(text=f"Best Tour: {best_tour}")
        self.distance_label.config(text=f"Distance: {current_distance}")

    def add_random_city(self):
        """Add a random city to the canvas."""
        alphabet = string.ascii_uppercase
        existing_cities = set(self.cities.keys())

        remaining_cities = [city for city in alphabet if city not in existing_cities]

        if not remaining_cities:
            messagebox.showinfo("No Cities to Add", "All cities (A-Z) have already been added. Please remove some cities to add new ones.")
            return

        city_name = remaining_cities[0]
        x = random.randint(50, 500)
        y = random.randint(50, 450)

        self.add_city(city_name, x, y)

    def add_city(self, city_name, x, y):
        """Add a new city to the list."""
        self.cities[city_name] = (x, y)
        self.draw_cities()

    def remove_city(self):
        """Remove a random city from the list."""
        if not self.cities:
            messagebox.showwarning("Warning", "No cities to remove.")
            return

        city_name = random.choice(list(self.cities.keys()))
        del self.cities[city_name]
        self.draw_cities()

    def draw_cities(self):
        """Redraw the cities and connections."""
        self.canvas.delete("all")

        for city, (x, y) in self.cities.items():
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="blue", width=0, tags=city)
            self.canvas.create_text(x, y - 15, text=city, font=("Arial", 10, "bold"), tags=city)

        for city1 in self.cities:
            for city2 in self.cities:
                if city1 != city2:
                    x1, y1 = self.cities[city1]
                    x2, y2 = self.cities[city2]
                    self.canvas.create_line(x1, y1, x2, y2, fill="gray", width=1, smooth=True)

if __name__ == "__main__":
    app = TSPGUI()
    app.mainloop()
