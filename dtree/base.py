import numpy as np
import pandas as pd

class Node:
    """
    Represents a node in the Decision Tree.
    """

    def __init__(self, depth=0, parent=None):
        self.depth = depth  # Distance from the root
        self.parent = parent  # Parent node
        self.feature_index = None  # Index of the feature used for splitting
        self.threshold = None  # Threshold value for splitting
        self.left_child = None  # Left child node
        self.right_child = None  # Right child node
        self.is_leaf = False  # True if it's a leaf node
        self.value = None  # Value (e.g., class label or mean value) if it's a leaf

        # Path information (complete path from root to this node)
        self.path_features = []
        self.path_thresholds = []

    def __str__(self):
        if self.is_leaf:
            return f"Leaf Node (Depth: {self.depth}, Value: {self.value})"
        else:
            return f"Decision Node (Depth: {self.depth}, Feature: {self.feature_index}, Threshold: {self.threshold})"

class DecisionTree:
    """
    A simple Decision Tree implementation for educational purposes.
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Maximum depth of the tree
        self.root = None  # Root node of the tree
        self.layers = []  # List to store nodes at each layer

    def _gini_impurity(self, y):
        """Calculates the Gini impurity of a set of labels."""
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1.0 - np.sum((counts / len(y)) ** 2)
        return impurity

    def _information_gain(self, y, y_left, y_right):
        """Calculates the information gain from a split."""
        p = len(y_left) / len(y)
        return self._gini_impurity(y) - p * self._gini_impurity(y_left) - (1 - p) * self._gini_impurity(y_right)

    def _best_split(self, X, y):
        """Finds the best feature and threshold for splitting."""
        best_feature = None
        best_threshold = None
        best_gain = 0

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                y_left = y[left_indices]
                y_right = y[~left_indices]

                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth, parent):
        """Recursively builds the decision tree."""
        node = Node(depth=depth, parent=parent)

        # Update path information
        if parent:
            node.path_features = parent.path_features + [parent.feature_index]
            node.path_thresholds = parent.path_thresholds + [parent.threshold]

        # Add node to the appropriate layer
        if depth >= len(self.layers):
            self.layers.append([])
        self.layers[depth].append(node)

        # Check for leaf conditions
        if depth == self.max_depth or len(np.unique(y)) == 1:
            node.is_leaf = True
            node.value = np.argmax(np.bincount(y))  # For classification (majority class)
            return node

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            node.is_leaf = True
            node.value = np.argmax(np.bincount(y))
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]

        # Recursively build the child nodes
        node.left_child = self._build_tree(X_left, y_left, depth + 1, node)
        node.right_child = self._build_tree(X_right, y_right, depth + 1, node)

        return node

    def fit(self, X, y):
        """Trains the decision tree."""
        self.root = self._build_tree(X, y, 0, None)

    # Add methods to explore the tree and get insights 
    # (e.g., print_tree, get_node_info, get_layer_info, get_path_info)
    # ... (Implementation of exploration methods)