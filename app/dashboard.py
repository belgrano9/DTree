import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image  # For displaying the image
from dtree.tree import DecisionTree
from dtree.utils import get_node_statistics, plot_nodesamples, plot_stacked
from sklearn.datasets import (
    make_moons,
)
from PIL import Image
from pdf2image import convert_from_path

# style to rag left/right


def main():
    # --- Create a Slightly More Complex Dataset ---
    X, y = make_moons(n_samples=100, noise=0.25, random_state=42)

    # --- Create the decision tree ---
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)

    # --- Visualize the tree ---
    tree.visualize_tree()  # Save the visualization to examples/decision_tree.pdf
    # images = convert_from_path("examples/decision_tree.pdf", dpi=300)
    # image = images[0]
    # --- Get node statistics ---
    depths, n_samples, class_counts = get_node_statistics(tree)

    # --- Streamlit Dashboard ---
    st.markdown(
        """
    <style>
        .block-container {
            padding: 0;
            margin: 0;
        }
        [data-testid="column"] {
            padding: 0;
            margin: 0;
        }
        [data-testid="stVerticalBlock"] {
            padding: 0;
            margin: 0;
        }
        .stImage > img {
            width: 100%;
            margin: 0;
            padding: 0;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.title("Decision Tree Dashboard")

    # --- Left column (image) ---
    col1, col2 = st.columns(2, gap="large")  # Create two columns
    with col1:
        st.markdown('<div class="ragged-left">', unsafe_allow_html=True)
        st.image("examples/decision_tree.svg", width=400)
        st.markdown("</div>", unsafe_allow_html=True)
        # st.image(image, width=400)  # Adjust width as needed

    # --- Right column (plots) ---
    with col2:
        # --- Node Number vs. Number of Samples ---
        fig1 = plot_nodesamples(n_samples)  # Get the figure object
        st.pyplot(fig1)  # Display the figure

        # --- Stacked Bar Chart of Class Counts ---
        fig2 = plot_stacked(class_counts)  # Get the figure object
        st.pyplot(fig2)


if __name__ == "__main__":
    main()
