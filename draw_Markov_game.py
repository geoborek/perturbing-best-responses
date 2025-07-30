import numpy as np
import matplotlib.pyplot as plt
from examples import Theorem3_2, Theorem3_3

g = Theorem3_2(16)
# g = Theorem3_3(16)

masks = g.masks

# RGB colors
colors = [
    (0, 0, 0),
    (1, 0, 0),    
    (0, 1, 0),    
    (0, 0, 1),    
    (1, 1, 0),    
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
    (0.5, 0.5, 0.5)

]

color_grid = np.zeros((16, 16, 3))

for mask, color in zip(masks, colors):
    for c in range(3):  # R, G, B channels
        color_grid[:, :, c] += mask * color[c]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')

for i in range(16):
    for j in range(16):
        rect = plt.Rectangle((j, 15 - i), 1, 1,
                             facecolor=color_grid[i, j],
                             edgecolor='black', linewidth=0.1)
        ax.add_patch(rect)

ax.set_xlim(0, 16)
ax.set_ylim(0, 16)

# Save as image
plt.savefig(f"{g}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{g}.pdf", bbox_inches='tight')  # for LaTeX
plt.close()