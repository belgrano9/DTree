import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from dtree.tree import DecisionTree
from dtree.utils import get_node_statistics, plot_nodesamples, plot_stacked

# Create a sample dataset (or load your own)
X = np.array([[1, 2], [2, 3], [3, 1], [4, 2]])
y = np.array([0, 1, 0, 1])

# Train the decision tree
tree = DecisionTree(max_depth=2)
tree.fit(X, y)

# Visualize the tree
tree.visualize_tree()

# Get node statistics
depths, n_samples, class_counts = get_node_statistics(tree)

plot_nodesamples(n_samples)

plot_stacked(class_counts)
