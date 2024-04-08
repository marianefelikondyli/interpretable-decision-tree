from scipy import stats

import numpy as np

from help_functions import is_numeric, class_counts
from performance_evaluation import accuracy, precision_and_recall, confusion_matrix, fscore


class Question:
    """Questions are used to partition a dataset."""

    def __init__(self, column, partition_values, partition_type, mode_of_each_column, header):
        self.column = column
        self.partition_values = partition_values
        self.partition_type = partition_type
        self.mode_of_each_column = mode_of_each_column
        self.header = header

    def match(self, row):
        """Answers question for a particular row (sample).
        Return index of which Child Branch to go."""

        if self.partition_type == "discrete":
            for t, val in enumerate(self.partition_values):
                if row[self.column] == val:
                    return t

            # if test data has a value not seen while training
            # unseen label is getting considered as mode label of that column
            for t, val in enumerate(self.partition_values):
                if self.mode_of_each_column[self.column] == val:
                    return t
            return 0

        # for continuous value, assign index of appropriate interval node child
        elif self.partition_type == "interval":
            if len(self.partition_values) == 1:
                return 0

            for k in range(1, len(self.partition_values) - 1):
                if (
                    self.partition_values[k - 1]
                    <= row[self.column]
                    < self.partition_values[k]
                ):
                    return k
            if row[self.column] >= self.partition_values[-2]:
                return len(self.partition_values) - 1
            if row[self.column] < self.partition_values[0]:
                return 0

    def __repr__(self):
        """Helper method to print question in readable form"""

        if self.partition_type == "discrete":
            return "which {} category ?".format(self.header[self.column])

        elif self.partition_type == "interval":
            return "which {} interval ?".format(self.header[self.column])

        equality_condition = "=="
        if is_numeric(self.value):
            if self.partition_type == "discrete":
                return "Is {} {} {} ?".format(
                    self.header[self.column], equality_condition, str(self.value)
                )
            elif self.partition_type == "interval":
                return "Is {} at range {} ?".format(
                    self.header[self.column], str(self.value)
                )

            return "Is {} {} {} ?".format(
                self.header[self.column], equality_condition, str(self.value)
            )


def entropy(Y):
    """Calculate entropy of data from labels"""
    labels, counts = class_counts(Y)
    probs = counts / float(Y.shape[0])
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def information_gain(y, intervals_indexes, current_entropy):
    """Information Gain:  The Entropy of the starting node, minus the weighted entropy of child nodes."""
    sum_ = 0
    len_sum = 0

    # weighted sum of entropies of each split (or interval)
    for interval_indices in intervals_indexes:
        sum_ += float(len(interval_indices) * entropy(y[interval_indices]))
        len_sum += len(interval_indices)

    return current_entropy - sum_ / len_sum


def gain_ratio(y, intervals_indexes, current_entropy):
    """Information_Gain divided by entropies of sub-dataset proportions."""

    # "split information" measure
    len_sum = 0
    for interval_indices in intervals_indexes:
        len_sum += len(interval_indices)

    sum_ = 0
    for interval_indices in intervals_indexes:
        sum_ += len(interval_indices) * np.log2(len_sum / len(interval_indices))
    split_information = sum_ / len_sum

    # prevent division by zero
    if split_information < 0.0001:
        split_information = 0.0001

    return information_gain(y, intervals_indexes, current_entropy) / split_information


""" ID3 Decision Tree Algorithm """


class Node:
    """Holds a reference to the question, and the child nodes that is partitioned by the question."""

    def __init__(self, child_branches, question, gain, y):
        self.child_branches = child_branches
        self.question = question
        self.gain = gain

        counts = np.column_stack(class_counts(y))
        self.label_counts = {row[0]: row[1] for row in counts}


class Leaf:
    """
    Leaf node contains a dictionary of counts of classes.
    Output is usually the majority class.
    """

    def __init__(self, y):
        # y passed as predictions dictionary while pruning the tree
        if type(y) is dict:
            self.predictions = y
        else:
            counts = np.column_stack(class_counts(y))
            self.predictions = {row[0]: row[1] for row in counts}


class Decision_Tree:
    def __init__(self, maximum_depth, header):
        self.depth = maximum_depth
        self.header = header

    def train(self, X_train, y_train, mode_of_each_column):
        # most_common_value_of_each_column = stats.mode(X_train)[0][0]  # is used when there is unseen labels inside test data
        self.mode_of_each_column = mode_of_each_column
        self.root = self._build_tree(X_train, y_train, self.depth, [])

    def find_best_split(self, X, y, used_attributes):
        """
        Find the best question to ask by iterating over every feature.
        Choose the best possible attribute and best possible split in terms of information gain.
        """
        copy_used_attributes = used_attributes[0:]
        best_gain = 0
        best_question = None
        best_partition = []
        used_attribute = 0
        current_entropy = entropy(y)
        n_features = X.shape[1]

        # To not use any attribute (to split the data-subset) twice, in a single path from root to leaf.
        for col in range(n_features):
            if col in copy_used_attributes:
                continue

            values = X[:, col]
            unique_values = np.unique(values)

            # Handling continuous features with percentile-based approach
            if is_numeric(values[0]):
                # Define percentiles to evaluate as potential splits
                # print("Len unique value", len(unique_values))
                if len(unique_values) == 1:
                        partition_to_indexesOfSamples = [X[:, col] <= unique_values[0]]
                        gain = information_gain(
                            y, partition_to_indexesOfSamples, current_entropy
                        )
                        if gain >= best_gain:
                            best_gain = gain
                            best_partition = partition_to_indexesOfSamples
                            best_question = Question(col, [unique_values[0]], "interval", self.mode_of_each_column, self.header)
                            used_attribute = col
                else:
                    percentiles = [np.percentile(unique_values, p) for p in range(10, 100, 10)]
                    unique_percentiles = np.unique(percentiles)
                    for split_value in unique_percentiles:
                        # print(split_value)
                        partitions_to_indexesOfSamples = [values <= split_value, values > split_value]
                        partitions_to_indexesOfSamples = [np.where(partition)[0] for partition in
                                                          partitions_to_indexesOfSamples]
                        # Calculate information gain or gain ratio for the split
                        gain = gain_ratio(y, partitions_to_indexesOfSamples, current_entropy)
                        # print(gain)

                        if gain > best_gain:
                            best_gain = gain
                            best_partition = partitions_to_indexesOfSamples
                            best_question = Question(col, [split_value], "interval", self.mode_of_each_column, self.header)
                            used_attribute = col


            # this column is a discrete attribute
            else:
                # multi-way split: as many branches as len(unique_values)
                unique_values = np.unique(X[:, col])
                partitions_to_indexesOfSamples = []

                for value in unique_values:
                    indexes_in_this_partition = X[:, col] == value
                    partitions_to_indexesOfSamples.append(indexes_in_this_partition)

                # gain = information_gain(y, partitions_to_indexesOfSamples, current_entropy)
                # gain /= len(unique_values)
                gain = gain_ratio(y, partitions_to_indexesOfSamples, current_entropy)

                if gain >= best_gain:
                    best_gain = gain
                    best_partition = partitions_to_indexesOfSamples
                    best_question = Question(col, unique_values, "discrete", self.mode_of_each_column, self.header)
                    used_attribute = col

        copy_used_attributes.append(used_attribute)
        return best_gain, best_partition, best_question, copy_used_attributes

    def _build_tree(self, X_train, y_train, depth, used_attributes):
        """recursive decision tree building function"""
        if X_train.shape[0] < 2:
            return Leaf(y_train)

        if depth < 1:
            return Leaf(y_train)

        all_samples_in_subset_are_same_label = np.all(y_train == y_train[0])
        if all_samples_in_subset_are_same_label:
            return Leaf(y_train)

        gain, partitions, question, used_attributes = self.find_best_split(
            X_train, y_train, used_attributes
        )

        if gain == 0 or question is None:
            return Leaf(y_train)

        child_branches = []
        for indices_in_this_partition in partitions:
            child_branches.append(
                self._build_tree(
                    X_train[indices_in_this_partition],
                    y_train[indices_in_this_partition],
                    depth - 1,
                    used_attributes,
                )
            )

        return Node(child_branches, question, gain, y_train)

    def predict(self, row):
        prediction = self._classify(row, self.root)
        return prediction

    def _classify(self, row, node):
        """Recursively traverses the decision tree towards a prediction(Leaf node)"""

        # Base case: At a leaf node
        if isinstance(node, Leaf):
            max_count = 0
            max_label = None

            for k, v in node.predictions.items():
                if int(v) >= max_count:
                    max_count = int(v)
                    max_label = k

            return max_label

        branch_to_go = node.child_branches[node.question.match(row)]
        return self._classify(row, branch_to_go)

    def test(self, X_test, y_test):
        """Returns accuracy and fscore of test dataset"""
        # Not vectorized.
        preds = []
        for i, row in enumerate(X_test):
            preds.append(self.predict(row))

        preds = np.array(preds)

        a = accuracy(preds, y_test)
        f = fscore(preds, y_test)
        p, r = precision_and_recall(preds, y_test)

        return a, f, p, r


def filter_interval(X, col, min_, m, interval_size):
    indices = []
    for j in range(np.shape(X)[0]):
        if (min_ + m * interval_size) <= X[j, col] < (min_ + (m + 1) * interval_size):
            indices.append(j)
    return indices


#  Visual Representation of Decision Tree


def print_tree(node, markerStr="+- ", levelMarkers=[]):
    level = len(levelMarkers)
    emptyStr = " " * 15
    connectionStr = "|" + emptyStr[:-4]
    mapper = lambda draw: connectionStr if draw else emptyStr
    markers = "".join(map(mapper, levelMarkers[:-1]))
    markers += markerStr if level > 0 else ""

    if isinstance(node, Leaf):
        print(f"{markers}Prediction (Class): {node.predictions}")
        return

    print(f"{markers}{node.question}")
    if node.question.partition_type == "discrete":
        for i, child in enumerate(node.child_branches):
            label_answer = "(" + str(node.question.partition_values[i]) + ") -- "
            isLast = i == len(node.child_branches) - 1
            print_tree(child, label_answer, [*levelMarkers, not isLast])

    else:
        for i, child in enumerate(node.child_branches):
            if i == 0:
                range_answer = (
                    "("
                    + " ,"
                    + format(node.question.partition_values[i], ".2f")
                    + ") -- "
                )
            elif i == len(node.child_branches) - 1:
                range_answer = (
                    "["
                    + format(node.question.partition_values[i - 1], ".2f")
                    + ", ) -- "
                )
            else:
                range_answer = (
                    "["
                    + format(node.question.partition_values[i - 1], ".2f")
                    + ","
                    + format(node.question.partition_values[i], ".2f")
                    + ") -- "
                )

            isLast = i == len(node.child_branches) - 1
            print_tree(child, range_answer, [*levelMarkers, not isLast])


