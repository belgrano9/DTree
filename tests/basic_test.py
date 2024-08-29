import unittest
import numpy as np
from dtree.tree import (
    DecisionTree,
)


class DecisionTreeTest(unittest.TestCase):
    def test_fit_predict_simple(self):
        # Create a simple dataset
        X = np.array([[1, 2], [2, 3], [3, 1], [4, 2]])
        y = np.array([0, 1, 0, 1])

        # Train the decision tree
        tree = DecisionTree(max_depth=2)
        tree.fit(X, y)

        # Make predictions
        X_test = np.array([[2.5, 2], [3.5, 3]])
        predictions = tree.predict(X_test)

        # Check predictions (adjust expected values based on your splitting criterion)
        self.assertEqual(predictions, [1, 1])

    def test_print_tree(self):
        # Create a simple dataset (or use a larger one if you prefer)
        X = np.array([[1, 2], [2, 3], [3, 1], [4, 2]])
        y = np.array([0, 1, 0, 1])

        # Train the decision tree
        tree = DecisionTree(max_depth=2)
        tree.fit(X, y)

        # Print the tree structure with statistics
        tree.print_tree()  # Call the print_tree method


if __name__ == "__main__":
    unittest.main()
