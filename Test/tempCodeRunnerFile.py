for i, height in enumerate(segment_heights):
#     # Plot each individual as a horizontal bar segment
#     ax.barh(1, height, left=bottom, color=colors[i], label=f'Individual {individuals[i+1]}')  # +1 because index starts at 0
#     bottom += height  # Update the left position for the next segment
#     # Annotate the segment with the individual number
#     ax.text(bottom - height / 2, 1, f'{individuals[i+1]}', ha='center', va='center', color='white', fontweight='bold'