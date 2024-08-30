from typing import List, Tuple, Dict
from dtree.node import Node
import matplotlib.pyplot as plt


def get_node_statistics(tree):
    """
    Traverse the tree and collect statistics for each node.

    Returns:
        Tuple[List[int], List[int], List[Dict[int, int]]]:
            A tuple containing lists of node depths, number of samples at each node,
            and class counts at each node.
    """
    depths = []
    n_samples = []
    class_counts = []

    def _traverse_and_collect(node, depth=0):
        depths.append(depth)
        n_samples.append(node.n_samples)
        class_counts.append(node.class_counts)

        if node.left:
            _traverse_and_collect(node.left, depth + 1)
        if node.right:
            _traverse_and_collect(node.right, depth + 1)

    _traverse_and_collect(tree.root)
    return depths, n_samples, class_counts


def create_stacked_bar_data(class_counts):
    """
    Create data for a stacked bar chart from class counts.

    Args:
        class_counts (List[Dict[int, int]]): List of class counts at each node.

    Returns:
        Tuple[List[int], List[int]]:
            A tuple containing lists of counts for class 0 and class 1 at each node.
    """
    class_0_counts = []
    class_1_counts = []
    for counts in class_counts:
        if counts is not None:  # Check if counts is not None
            class_0_counts.append(counts[0] if 0 in counts else 0)
            class_1_counts.append(counts[1] if 1 in counts else 0)
        else:
            class_0_counts.append(0)
            class_1_counts.append(0)
    return class_0_counts, class_1_counts


def plot_nodesamples(n_samples):
    plt.figure()
    plt.plot(range(len(n_samples)), n_samples, marker="o")
    plt.xlabel("Node Number")
    plt.ylabel("Number of Samples")
    plt.title("Node Number vs. Number of Samples")
    plt.show()


def plot_stacked(class_counts):
    class_0_counts, class_1_counts = create_stacked_bar_data(class_counts)
    plt.figure()
    plt.bar(range(len(class_0_counts)), class_0_counts, label="Class 0")
    plt.bar(
        range(len(class_1_counts)),
        class_1_counts,
        bottom=class_0_counts,
        label="Class 1",
    )
    plt.xlabel("Node Number")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution at Each Node")
    plt.legend()
    plt.show()
