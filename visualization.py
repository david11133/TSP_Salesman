import tkinter as tk
from tkinter import messagebox, ttk
import random
import string
from tsp_solver import TSPSolver

# Constants
WINDOW_DIMENSIONS = {"width": 800, "height": 600}
BUTTON_SIZE = {"width": 30, "height": 1}
COLORS = {
    "control_bg": "#2C3E50",  # Dark Slate Blue (Modern, sleek background for control panel)
    "canvas_bg": "#F4F6F7",   # Light Grayish White (Subtle and soft background for the canvas)
    "best_tour": "#EAF0F1",    # Very Light Gray (Light contrast for the best tour text)
    "distance": "#EAF0F1",     # Very Light Gray (Light contrast for the distance text)
    "btn_start": "#3498DB",    # Vivid Blue (Clean, modern button for start)
    "btn_add": "#F39C12",      # Vibrant Yellow Orange (Eye-catching add city button)
    "btn_remove": "#E74C3C",   # Soft Red (Intuitive warning color for remove city)
    "node": "#16A085",         # Soft Teal (Modern, calm, and easy on the eyes for nodes)
    "edge": "#D5DBDB",         # Light Silver Gray (Subtle edges for a clean look)
    "highlight_node": "#FF5733",  # Warm Coral (Bright and noticeable for highlighting nodes)
    "highlight_edge": "#1ABC9C"   # Vibrant Mint Green (Clear, fresh color for highlighted edges)
}


DEFAULTS = {
    "population_size": 120,
    "generations": 20,
    "mutation_rate": 0.1
}


class TSPGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSP with Genetic Algorithm")
        self.geometry(f"{WINDOW_DIMENSIONS['width']}x{WINDOW_DIMENSIONS['height']}")
        self.cities = {}
        self.solver = TSPSolver(update_callback=self.update_stats)
        self._initialize_ui()

    def _initialize_ui(self):
        """Setup the layout and the different sections of the GUI."""
        self._create_frames()
        self._create_canvas()
        self._create_controls()
        self._create_buttons()

    def _create_frames(self):
        """Create main UI frames for control panel and canvas."""
        self.control_frame = tk.Frame(self, bg=COLORS["control_bg"])
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _create_canvas(self):
        """Create the canvas for city nodes and their connections."""
        self.canvas = tk.Canvas(self.canvas_frame, bg=COLORS["canvas_bg"], scrollregion=(0, 0, 1000, 1000))
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._add_canvas_scrollbars()

    def _add_canvas_scrollbars(self):
        """Add vertical and horizontal scrollbars for canvas."""
        self.canvas_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas_scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.config(yscrollcommand=self.canvas_scrollbar.set)

        self.canvas_hscrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas_hscrollbar.grid(row=1, column=0, sticky="ew")
        self.canvas.config(xscrollcommand=self.canvas_hscrollbar.set)

        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

    def _create_controls(self):
        """Create the statistics labels and input fields."""
        self._create_stats_labels()
        self._create_input_fields()

    def _create_stats_labels(self):
        """Create labels for displaying the best tour and distance."""
        stats_frame = tk.Frame(self.control_frame, bg=COLORS["control_bg"])
        stats_frame.pack(fill=tk.Y, padx=5, pady=10)

        self.best_tour_label = self._create_label(stats_frame, "Best Tour: ")
        self.distance_label = self._create_label(stats_frame, "Distance: ")

    def _create_label(self, parent_frame, text):
        """Helper function to create labels with common settings."""
        label = tk.Label(parent_frame, text=text, anchor="w", wraplength=150, bg=COLORS["control_bg"], fg=COLORS["best_tour"], font=("Arial", 12, "bold"))
        label.grid(sticky="w", padx=5, pady=5)
        return label

    def _create_input_fields(self):
        """Create input fields for population size, number of generations, and mutation rate."""
        input_frame = tk.Frame(self.control_frame, bg=COLORS["control_bg"])
        input_frame.pack(fill=tk.Y, padx=5, pady=(80, 10))

        self.pop_size_entry = self._create_input_field(input_frame, "Population Size:", DEFAULTS["population_size"])
        self.gen_entry = self._create_input_field(input_frame, "Number of Generations:", DEFAULTS["generations"])
        self.mutation_rate_entry = self._create_input_field(input_frame, "Mutation Rate:", DEFAULTS["mutation_rate"])

    def _create_input_field(self, parent_frame, label_text, default_value):
        """Helper function to create labeled input fields."""
        label = tk.Label(parent_frame, text=label_text, bg=COLORS["control_bg"], fg="white", font=("Arial", 10))
        label.grid(sticky="w", padx=5, pady=5)
        entry = tk.Entry(parent_frame, bg="#ffffff", fg="black")
        entry.grid(sticky="w", padx=5, pady=5)
        entry.insert(0, str(default_value))
        return entry

    def _create_buttons(self):
        """Create control buttons."""
        buttons_frame = tk.Frame(self.control_frame, bg=COLORS["control_bg"])
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self._create_button(buttons_frame, "Start", self.run_genetic_algorithm, COLORS["btn_start"])
        self._create_button(buttons_frame, "Add City", self.add_random_city, COLORS["btn_add"])
        self._create_button(buttons_frame, "Remove City", self.remove_city, COLORS["btn_remove"])

    def _create_button(self, parent_frame, text, command, color):
        """Helper function to create buttons."""
        button = tk.Button(parent_frame, text=text, command=command, width=BUTTON_SIZE["width"], height=BUTTON_SIZE["height"], bg=color, fg="white")
        button.pack(pady=10)

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

    def _validate_inputs(self, population_size, generations, mutation_rate):
        """Validate the user inputs for population size, generations, and mutation rate."""
        if population_size <= 0 or generations <= 0 or not (0 <= mutation_rate <= 1):
            messagebox.showerror("Invalid Input", "Please enter valid values for population size, generations, and mutation rate.")
            return False

        if population_size < len(self.cities):
            messagebox.showerror("Invalid Population Size", "Population size must be greater than or equal to the number of cities.")
            return False
        return True

    def update_stats(self, best_tour, current_distance, generation):
        """Update the labels with the best tour and its distance."""
        self.best_tour_label.config(text=f"Best Tour: {best_tour}")
        self.distance_label.config(text=f"Distance: {current_distance}")

    def add_random_city(self):
        """Add a random city to the map."""
        city_name = self._generate_random_city_name()
        x, y = self._generate_random_position()

        self.add_city(city_name, x, y)

    def _generate_random_city_name(self):
        """Generate a unique city name not yet in the list."""
        alphabet = string.ascii_uppercase
        used_cities = set(self.cities.keys())
        available_cities = [city for city in alphabet if city not in used_cities]
        
        if not available_cities:
            messagebox.showinfo("No Cities to Add", "All cities (A-Z) have been added. Please remove some cities.")
            return None
        
        return available_cities[0]

    def _generate_random_position(self):
        """Generate a random (x, y) position for the city."""
        x = random.randint(50, 500)
        y = random.randint(50, 450)
        return x, y

    def add_city(self, city_name, x, y):
        """Add a new city to the city list."""
        self.cities[city_name] = (x, y)
        self.draw_cities()

    def remove_city(self):
        """Remove a random city."""
        if self.cities:
            city_name = random.choice(list(self.cities.keys()))
            del self.cities[city_name]
            self.draw_cities()
        else:
            messagebox.showwarning("Warning", "No cities to remove.")

    def draw_cities(self):
        """Draw cities and their connections."""
        self.canvas.delete("all")

        for city, (x, y) in self.cities.items():
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=COLORS["node"], width=0, tags=city)
            self.canvas.create_text(x, y - 15, text=city, font=("Arial", 10, "bold"), tags=city)

        self._draw_edges()

    def _draw_edges(self):
        """Draw edges between all cities."""
        for city1, (x1, y1) in self.cities.items():
            for city2, (x2, y2) in self.cities.items():
                if city1 != city2:
                    self.canvas.create_line(x1, y1, x2, y2, fill=COLORS["edge"], width=1, smooth=True)


if __name__ == "__main__":
    app = TSPGUI()
    app.mainloop()
