import math

def calculate_distance(cities, tour):
    """Calculate the total distance of a tour."""
    return sum(distance(cities, city1, city2) for city1, city2 in zip(tour, tour[1:]))

def distance(cities, city1, city2):
    """Calculate the Euclidean distance between two cities."""
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def draw_path(canvas, cities, tour, color="blue"):
    """Draw the path of the tour on the canvas."""
    if canvas is not None:
        canvas.delete(f"path_{color}")  # Clear any existing path
        for i in range(len(tour)):
            city1 = tour[i]
            city2 = tour[(i + 1) % len(tour)]  # Wrap around to first city
            x1, y1 = cities[city1]
            x2, y2 = cities[city2]
            canvas.create_line(
                x1, y1, x2, y2, fill=color, tags=f"path_{color}", width=2, smooth=True
            )

def best_solution(population, fitness):
    """Return the best solution from the population."""
    best_index = max(range(len(fitness)), key=lambda i: fitness[i])
    return population[best_index]

def get_best_solution(cities, population):
    """Get the best solution and its distance."""
    best_tour = min(population, key=lambda tour: calculate_distance(cities, tour))
    best_distance = calculate_distance(cities, best_tour)
    return best_tour, best_distance

def update_visuals(canvas, root, update_callback, cities, best_tour, best_distance):
    """Update the visual components and labels."""
    
    # Ensure best_tour starts and ends with the same city
    if best_tour[0] != best_tour[-1]:
        # Convert best_tour to a list, append the first city to the end, then convert it back to a tuple if needed
        best_tour = list(best_tour) + [best_tour[0]]  # Append the first city to the end if not already

    # Draw the best tour so far
    draw_path(canvas, cities, best_tour, color="blue")  # Show the best path so far

    # If you have a callback to update the visuals, call it
    if update_callback:
        update_callback(best_tour, best_distance)

    # Update the labels for best tour and best distance
    if root:
        best_tour_label = root.nametowidget("best_tour_label")
        best_distance_label = root.nametowidget("best_distance_label")

        # Update the best tour label with the corrected tour
        best_tour_label.config(text=f"Best Tour: {' -> '.join(best_tour)}")  # Display the best tour as a readable string

        best_distance_label.config(text=f"Best Distance: {best_distance:.2f}")
