"""
Implementation from scratch of DecisionTree. 

This file contains the tree class.
"""

from typing import List, Tuple, Dict, Any
from .node import Node
from collections import Counter
import numpy as np  # Make sure to import numpy
import graphviz


class DecisionTree:
    def __init__(self, max_depth: int = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Node = None

    def _gini(self, y: np.ndarray) -> float:
        """Calculate the Gini impurity for a set of labels."""
        counts = Counter(y)
        impurity = 1.0
        for count in counts.values():
            prob_of_lbl = count / len(y)
            impurity -= prob_of_lbl**2
        return impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """
        Find the best feature and threshold to split the data based on Gini impurity.

        Returns:
            Tuple[int, float]: A tuple containing the index of the best feature to split on
                              and the best threshold value.
        """
        best_gain = -1.0  # Initialize with a negative value
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            thresholds = sorted(
                set(X[:, feature_idx])
            )  # Get unique values as potential thresholds

            for threshold in thresholds:
                # Split data based on the threshold
                left_indices = X[:, feature_idx] <= threshold
                right_indices = X[:, feature_idx] > threshold

                # Calculate Gini impurity for the split
                gini_left = self._gini(y[left_indices])
                gini_right = self._gini(y[right_indices])

                # Calculate weighted Gini impurity
                n_left = len(y[left_indices])
                n_right = len(y[right_indices])
                weighted_gini = (n_left / n_samples) * gini_left + (
                    n_right / n_samples
                ) * gini_right

                # Calculate information gain
                info_gain = self._gini(y) - weighted_gini

                # Update best split if the current split is better
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.

        Returns:
            Node: The root node of the built subtree.
        """
        n_samples, n_features = X.shape

        # Base cases for stopping recursion
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(set(y)) == 1
        ):  # All samples belong to the same class

            # Create a leaf node (Corrected part)
            leaf_value = Counter(y).most_common(1)[0][0]  # Most frequent class
            return Node(
                value=leaf_value,
                n_samples=n_samples,  # Set n_samples for leaf nodes
                class_counts=Counter(y),  # Set class_counts for leaf nodes
            )

        # Find the best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)

        # Create a decision node
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        node_gini = self._gini(y)
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            n_samples=n_samples,
            class_counts=Counter(y),
            gini_impurity=node_gini,
            info_gain=best_gain,  # Pass best_gain to the Node constructor
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build the decision tree from the training data."""
        self.root = self._build_tree(X, y)

    def predict(self, X: np.ndarray) -> List[Any]:
        """
        Traverse the tree to make predictions for new data points.

        Returns:
            List[Any]: A list of predictions, one for each data point in X.
        """
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x: np.ndarray, node: Node) -> Any:
        """
        Recursively traverse the tree to find the leaf node for a data point.

        Returns:
            Any: The predicted value (class label) for the data point.
        """
        if node.value is not None:  # Reached a leaf node
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def print_tree(self, node=None, depth=0):
        """Print the tree structure with statistics."""
        if node is None:
            node = self.root

        if node.value is not None:  # Leaf node
            print(
                "  " * depth
                + f"Leaf: Class={node.value}, Samples={node.n_samples}, Counts={node.class_counts}"
            )
        else:  # Decision node
            print(
                "  " * depth
                + f"Decision: Feature={node.feature}, Threshold={node.threshold}, "
                f"Samples={node.n_samples}, Counts={node.class_counts}, Gini={node.gini_impurity:.3f}, Gain={node.info_gain:.3f}"
            )
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

    def visualize_tree(self, filename="decision_tree"):
        """Visualize the decision tree using graphviz."""
        dot = graphviz.Digraph(comment="Decision Tree")
        node_counter = 0  # Initialize node counter

        def _visualize_node(node, node_id):
            nonlocal node_counter  # Access the outer node_counter

            if node is None:  # Check if node is None (base case)
                return

            node_counter += 1
            current_node_id = str(node_counter)  # Use node counter as ID

            if node.value is not None:  # Leaf node
                label = f"Node: {current_node_id}\nClass: {node.value}\nSamples: {node.n_samples}\n"
                for cls, count in node.class_counts.items():
                    percentage = (count / node.n_samples) * 100
                    label += f"Class {cls}: {percentage:.1f}%\n"
                dot.node(current_node_id, label)
            else:  # Decision node
                label = f"Node: {current_node_id}\nFeature: {node.feature}\nThreshold: {node.threshold}\nSamples: {node.n_samples}\n"
                for cls, count in node.class_counts.items():
                    percentage = (count / node.n_samples) * 100
                    label += f"Class {cls}: {percentage:.1f}%\n"
                label += f"Gini: {node.gini_impurity:.3f}\nGain: {node.info_gain:.3f}"
                dot.node(current_node_id, label)

                # Recursively visualize left and right children
                if node.left:
                    left_child_id = _visualize_node(node.left, current_node_id + "L")
                    dot.edge(current_node_id, left_child_id, label="True")
                if node.right:
                    right_child_id = _visualize_node(node.right, current_node_id + "R")
                    dot.edge(current_node_id, right_child_id, label="False")

            return current_node_id  # Return the current node's ID

        _visualize_node(self.root, "Root")
        dot.render(f"examples/{filename}", format="svg", view=False)
