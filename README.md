
# DTree: Interpretable Decision Tree Implementation

This repository contains a Python implementation of a decision tree algorithm from scratch, with a focus on interpretability and explainability.

**Key Features:**

- **Interpretability:** The code is designed to be easy to understand and follow, making it ideal for educational purposes.
- **Explainability:** The implementation provides methods for visualizing the decision tree structure and collecting statistics at each node, allowing you to understand how the tree is making decisions.
- **Streamlit Dashboard:**  A Streamlit dashboard is included to interactively visualize the decision tree and its statistics.

**Purpose:**

The primary purpose of this project is to demonstrate the inner workings of decision trees in a clear and understandable way. The code is designed to be used as a learning resource, allowing you to:

- Understand the core concepts of decision trees, such as splitting criteria, impurity measures (Gini impurity), information gain, and tree traversal.
- Explore how different tree parameters (like maximum depth) affect the tree's structure and performance.
- Visualize the decision boundaries and decision-making process of the tree.
- Collect and analyze statistics at each node to gain insights into the data and the tree's behavior.

**Use with Substack Post:**

This repository is specifically designed to be used as a companion to a Substack post explaining decision trees. The code examples, visualizations, and dashboard can be embedded or referenced in the Substack post to provide an interactive and engaging learning experience for readers.

**Getting Started:**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/belgrano9/DTree.git
   ```
2. **Install dependencies:**

   ```bash
   cd DTree
   poetry install
   ```
3. **Run the Streamlit dashboard:**

   ```bash
   poetry run run-dashboard 
   ```

**File Structure:**

- `dtree/`: Contains the Python modules for the decision tree implementation.
  - `node.py`: Defines the `Node` class, representing a node in the decision tree.
  - `tree.py`: Contains the `DecisionTree` class, which implements the decision tree algorithm.
  - `utils.py`: Helper functions for calculating statistics and creating visualizations.
- `examples/`: Contains example scripts and the Streamlit dashboard script (`dashboard.py`).
- `tests/`: Contains unit tests for the decision tree implementation.

**Contributing:**

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

**License:**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
