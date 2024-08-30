import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import (
    make_moons,
)  # Use make_moons for a slightly more complex dataset
from dtree.tree import DecisionTree
from dtree.utils import get_node_statistics, plot_nodesamples, plot_stacked

# --- Create a Slightly More Complex Dataset ---
X, y = make_moons(n_samples=100, noise=0.25, random_state=42)

# --- Train the decision tree ---
tree = DecisionTree(max_depth=3)
tree.fit(X, y)

# --- Visualize the tree ---
tree.visualize_tree()

# --- Get node statistics ---
depths, n_samples, class_counts = get_node_statistics(tree)

plot_nodesamples(n_samples)

plot_stacked(class_counts)
