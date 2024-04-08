import numpy as np


class Node:
    def __init__(self, feature_index=None, threshold=None, value=None, children=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold  # Threshold value for feature split
        self.value = value  # Prediction value if node is leaf
        self.children = children if children else []


class MultiwayDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_children=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_children = max_children

    def _split_dataset(self, X, y, feature_index, threshold):
        masks = [X.iloc[:, [feature_index]] <= threshold]
        thresholds = [threshold]
        unique_values = np.unique(X.iloc[:, [feature_index]])
        if len(unique_values) > self.max_children:
            threshold_step = (unique_values.max() - unique_values.min()) / self.max_children
            for i in range(1, self.max_children):
                thresholds.append(unique_values.min() + i * threshold_step)
                masks.append(np.logical_and(X.iloc[:, [feature_index]] > thresholds[i - 1], X.iloc[:, [feature_index]] <= thresholds[i]))
        masks.append(X.iloc[:, [feature_index]] > thresholds[-1])
        return masks

    def _calculate_gini(self, y):
        classes = np.unique(y)
        n_samples = len(y)
        gini = 1.0
        for c in classes:
            p = np.sum(y == c) / n_samples
            gini -= p ** 2
        return gini

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None
        n_features = X.shape[1]
        print(f"Number of features: {n_features}")

        for feature_index in range(n_features):
            print(feature_index)
            print(X)
            print(type(X))
            thresholds = np.unique(X.iloc[:, [feature_index]])
            print(thresholds)
            for threshold in thresholds:
                masks = self._split_dataset(X, y, feature_index, threshold)
                gini = 0
                for mask in masks:
                    print(mask)
                    y_splitted = y[mask]
                    gini += len(y_splitted) / len(y) * self._calculate_gini(y_splitted)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth=0):
        print(depth)
        if len(np.unique(y)) == 1:  # If all samples belong to the same class
            return Node(value=y[0])

        if depth == self.max_depth:
            return Node(value=np.mean(y))

        if len(y) < self.min_samples_split:
            return Node(value=np.mean(y))

        best_feature_index, best_threshold = self._find_best_split(X, y)

        if best_feature_index is None:
            return Node(value=np.mean(y))

        masks = self._split_dataset(X, y, best_feature_index, best_threshold)

        children = []
        for mask in masks:
            X_child = X[mask]
            y_child = y[mask]
            if len(X_child) < self.min_samples_leaf:
                children.append(Node(value=np.mean(y_child)))
            else:
                children.append(self._build_tree(X_child, y_child, depth + 1))

        return Node(feature_index=best_feature_index, threshold=best_threshold, children=children)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        for i, threshold in enumerate(node.thresholds):
            if x[node.feature_index] <= threshold:
                return self._predict_one(x, node.children[i])
        return self._predict_one(x, node.children[-1])

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict_one(x, self.root))
        return np.array(predictions)


def print_tree(node, depth=0, indent="  "):
    if node is None:
        return

    if node.value is not None:
        print(depth * indent + "Predict:", node.value)
        return

    print(depth * indent + "Feature:", node.feature_index)
    print(depth * indent + "Threshold:", node.threshold)

    print(depth * indent + "Left:")
    print_tree(node.left, depth + 1, indent)

    print(depth * indent + "Right:")
    print_tree(node.right, depth + 1, indent)

