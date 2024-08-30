import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image  # For displaying the image
from dtree.tree import DecisionTree
from dtree.utils import get_node_statistics, plot_nodesamples, plot_stacked
from sklearn.datasets import (
    make_moons,
)

# --- Create a Slightly More Complex Dataset ---
X, y = make_moons(n_samples=100, noise=0.25, random_state=42)

# --- Create the decision tree ---
tree = DecisionTree(max_depth=3)
tree.fit(X, y)

# --- Visualize the tree ---
tree.visualize_tree()  # Save the visualization to examples/decision_tree.pdf

# --- Get node statistics ---
depths, n_samples, class_counts = get_node_statistics(tree)

# --- Streamlit Dashboard ---
st.title("Decision Tree Dashboard")

# --- Left column (image) ---
col1, col2 = st.columns(2)  # Create two columns
with col1:
    image = Image.open("examples/decision_tree.pdf")
    st.image(image, width=400)  # Adjust width as needed

# --- Right column (plots) ---
with col2:
    # --- Node Number vs. Number of Samples ---
    fig1 = plot_nodesamples(n_samples)

    # --- Stacked Bar Chart of Class Counts ---

    fig2 = plot_stacked(class_counts)
    st.pyplot(fig2)
