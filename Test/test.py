import matplotlib.pyplot as plt
import numpy as np

# Data for the chart
individuals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cumulative_probabilities = [0, 0.0858, 0.1860, 0.3063, 0.3921, 0.4923, 0.5496, 0.6128, 0.7636, 0.8494, 1.0]

# Calculate the heights of each segment (each individual) based on cumulative probabilities
segment_heights = np.diff(cumulative_probabilities)

# Assigning colors to each segment (one color per individual)
colors = plt.cm.get_cmap("tab10", len(individuals))(np.linspace(0, 1, len(individuals)))

# Plotting the stacked bar chart horizontally
fig, ax = plt.subplots(figsize=(8, 4))  # Adjusted for a thinner, horizontal bar

bottom = 0  # Start from 0 for the first individual
for i, height in enumerate(segment_heights):
    # Plot each individual as a horizontal bar segment
    ax.barh(1, height, left=bottom, color=colors[i])  # 'left' controls where each individual starts
    bottom += height  # Update the left position for the next segment
    # Annotate the segment with the individual number
    ax.text(bottom - height / 2, 1, f'{individuals[i+1]}', ha='center', va='center', color='white', fontweight='bold')

# Add labels and title
ax.set_title("Stacked Bar Chart with One Horizontal Bar for Individuals")
ax.set_xlabel("Cumulative Probability")
ax.set_yticks([1])  # Only one bar, so y-tick is at position 1
ax.set_yticklabels(["All Individuals"])

# Display the plot
plt.show()
