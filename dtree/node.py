class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        stats=None,
        n_samples=None,
        class_counts=None,
        gini_impurity=None,
        info_gain=None,
    ):

        self.feature = feature  # Feature index used for splitting
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left subtree (for values <= threshold)
        self.right = right  # Right subtree (for values > threshold)
        self.value = value  # Class label (for leaf nodes)
        self.stats = stats  # Dictionary to store node statistics

        self.n_samples = n_samples  # Number of samples at the node
        self.class_counts = class_counts  # Class distribution at the node
        self.gini_impurity = gini_impurity  # Gini impurity at the node
        self.info_gain = info_gain  # Information gain from the split (if applicable)
