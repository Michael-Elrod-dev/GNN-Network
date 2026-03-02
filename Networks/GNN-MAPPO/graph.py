import matplotlib.pyplot as plt
import numpy as np

# Data points (estimated from graph)
agents = [2, 4, 8, 15, 25, 35]

dqn_250k = [95, 82, 77, 88, 80, 95]
dqn_1mil = [96, 90, 85, 80, 82, 85]
gnn_250k = [99, 95, 97, 99, 99, 99]
gnn_1mil = [100, 98, 98, 100, 100, 100]

x = np.arange(len(agents))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 6))

gap = 0.05  # adjust this value to increase/decrease spacing

bars1 = ax.bar(x - 1.5 * width - gap / 2, dqn_250k, width, label="DQN-250k", color="steelblue")
bars2 = ax.bar(x - 0.5 * width - gap / 2, dqn_1mil, width, label="DQN-1mil", color="mediumseagreen")
bars3 = ax.bar(x + 0.5 * width + gap / 2, gnn_250k, width, label="GNN-250k", color="palevioletred")
bars4 = ax.bar(x + 1.5 * width + gap / 2, gnn_1mil, width, label="GNN-1mil", color="mediumpurple")

# Axes
ax.set_ylim(40, 105)
ax.set_yticks([40, 60, 80, 100])
ax.set_xticks(x)
ax.set_xticklabels(agents)

ax.set_xlabel("Number of Agents", fontsize=12)
ax.set_ylabel("Grid Coverage (%)", fontsize=12)

# Grid
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

# Legend below the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=10, frameon=True)

# Border
for spine in ax.spines.values():
    spine.set_visible(True)

plt.tight_layout()
plt.savefig("bar_graph.png", dpi=150, bbox_inches="tight")
plt.show()
