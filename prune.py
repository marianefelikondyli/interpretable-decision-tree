import copy

import pandas as pd
from sklearn.model_selection import train_test_split

from decision_tree import Leaf, Decision_Tree, print_tree




""" Preprocessing """

df = pd.read_csv("./WA_Fn-UseC_-HR-Employee-Attrition.csv")
pd.options.display.max_columns = len(df.columns)
pd.set_option("display.precision", 6)
header = df.drop(["Attrition"], axis=1).columns
ground_truth_label = "Attrition"
ground_truth_classes = df.Attrition.unique()

## Number of occurences of class types
print(df.Attrition.value_counts())
print("\n\n")

X = df.drop(["Attrition"], axis=1)
y = df.Attrition
mode_of_each_column = X.mode().iloc[0].to_list()
print(mode_of_each_column)

X = X.to_numpy()
Y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=24, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=24, shuffle=True
)


""" Pruning methods: """


def _find_twigs(Node, twigs):
    if isinstance(Node, Leaf):
        return
    current_is_twig = True
    for child in Node.child_branches:
        if not isinstance(child, Leaf):
            _find_twigs(child, twigs)
            current_is_twig = False

    if current_is_twig:
        twigs.append(Node)


def find_twig_with_least_info_gain(root_):
    twigs = []
    _find_twigs(root_, twigs)

    if len(twigs) == 0:
        return None
    min_gain = twigs[0].gain
    min_index = 0

    for i, twig in enumerate(twigs):
        if twig.gain < min_gain:
            min_gain = twig.gain
            min_index = i
    return twigs[i], len(twigs)


def remove_twig_from_tree(node, twig):
    if isinstance(node, Leaf) or node is None:
        None
    else:
        for i, child in enumerate(node.child_branches):
            if child is twig:
                node.child_branches[i] = Leaf(child.label_counts)
                print(
                    "pruned twig predictions: "
                    + str(node.child_branches[i].predictions)
                )
                break
            else:
                remove_twig_from_tree(child, twig)


def remove_twig_from_model(model, twig):
    remove_twig_from_tree(model.root, twig)


def prune_by_least_info_gain(model, X_val, y_val):
    # recursive function, but "model" does not get assigned deeper nodes in the tree, stays at root level

    prev_acc, prev_f, prev_p, prev_r = model.test(X_val, y_val)

    modified_model = copy.deepcopy(model)
    twig, twig_count = find_twig_with_least_info_gain(modified_model.root)

    if twig == modified_model.root:
        return model
    remove_twig_from_model(modified_model, twig)

    acc, f, p, r = modified_model.test(X_val, y_val)

    print(
        "Amount of twigs at tree {} \t\tOld acc {}, new acc {}\n".format(
            twig_count, prev_acc, acc
        )
    )

    if acc >= prev_acc:
        return prune_by_least_info_gain(modified_model, X_val, y_val)
    return model


"""Pruning Decision Tree with 4 maximum_depth"""

# First, create a Decision Tree based on Training set.Then prune it.
model = Decision_Tree(4)
model.train(X_train, y_train, mode_of_each_column)
print("PRUNING MAX_DEPTH = 4 TREE: \n")
pruned_model = prune_by_least_info_gain(model, X_val, y_val)


#### Comparison of Before and After Pruning:

print("\nBEFORE PRUNING:  (MAX_DEPTH = 4 TREE)\n")
print_tree(model.root)
print("\n\n")
print("AFTER PRUNING:   (MAX_DEPTH = 4 TREE)\n")
print_tree(pruned_model.root)
print("\n")


#### Comparing Before and After Pruning Scores:
#
# print("BEFORE PRUNING  (MAX_DEPTH = 4 TREE) : ")
# train_acc, train_f, train_p, train_r = model.test(X_train, y_train)
# print("  Train :      F1 Score: {:.3f},   Accuracy: {:.3f}".format(train_f, train_acc))
# val_acc, val_f, val_p, val_r = model.test(X_val, y_val)
# print("  VALIDATION : F1 Score: {:.3f} ,  Accuracy: {:.3f}".format(val_f, val_acc))
# tes_acc, tes_f, tes_p, tes_r = model.test(X_test, y_test)
# print("  TEST :       F1 Score: {:.3f} ,  Accuracy: {:.3f} \n".format(tes_f, tes_acc))
#
# print("AFTER PRUNING  (MAX_DEPTH = 4 TREE) :")
# train_acc, train_f, train_p, train_r = pruned_model.test(X_train, y_train)
# print("  Train :      F1 Score: {:.3f},   Accuracy: {:.3f}".format(train_f, train_acc))
# val_acc, val_f, val_p, val_r = pruned_model.test(X_val, y_val)
# print("  VALIDATION : F1 Score: {:.3f} ,  Accuracy: {:.3f}".format(val_f, val_acc))
# tes_acc, tes_f, tes_p, tes_r = pruned_model.test(X_test, y_test)
# print("  TEST :       F1 Score: {:.3f} ,  Accuracy: {:.3f} \n".format(tes_f, tes_acc))
#
#
# """ Pruning Decision Tree with 15 maximum_depth"""
#
# # First, create a Decision Tree based on Training set.Then prune it.
# model = Decision_Tree(15)
# model.train(X_train, y_train, mode_of_each_column)
# print("\n\n\n\nPRUNING MAX_DEPTH = 15 TREE: \n")
# pruned_model = prune_by_least_info_gain(model, X_val, y_val)
#
#
# #### Comparison of Before and After Pruning:
#
# print("\nBEFORE PRUNING  (MAX_DEPTH = 15 TREE) :\n")
# print(
#     "This tree contains very long text lines and might not print well, due to line wrapping settings. Output exceeds size limit.\n"
# )
# print_tree(model.root)
# print("\n\nAFTER PRUNING  (MAX_DEPTH = 15 TREE) :\n")
# print(
#     "This tree contains very long text lines and might not print well, due to line wrapping settings. Output exceeds size limit.\n"
# )
# print_tree(pruned_model.root)
# print("\n")
#
#
# #### Comparing Before and After Pruning Scores:
#
# print("\nBEFORE PRUNING  (MAX_DEPTH = 15 TREE) :")
# train_acc, train_f, train_p, train_r = model.test(X_train, y_train)
# print("  Train :      F1 Score: {:.3f},   Accuracy: {:.3f}".format(train_f, train_acc))
# val_acc, val_f, val_p, val_r = model.test(X_val, y_val)
# print("  VALIDATION : F1 Score: {:.3f} ,  Accuracy: {:.3f}".format(val_f, val_acc))
# tes_acc, tes_f, tes_p, tes_r = model.test(X_test, y_test)
# print("  TEST :       F1 Score: {:.3f} ,  Accuracy: {:.3f} \n\n".format(tes_f, tes_acc))
#
# print("\nAFTER PRUNING  (MAX_DEPTH = 15 TREE) :")
# train_acc, train_f, train_p, train_r = pruned_model.test(X_train, y_train)
# print("  Train :      F1 Score: {:.3f},   Accuracy: {:.3f}".format(train_f, train_acc))
# val_acc, val_f, val_p, val_r = pruned_model.test(X_val, y_val)
# print("  VALIDATION : F1 Score: {:.3f} ,  Accuracy: {:.3f}".format(val_f, val_acc))
# tes_acc, tes_f, tes_p, tes_r = pruned_model.test(X_test, y_test)
# print("  TEST :       F1 Score: {:.3f} ,  Accuracy: {:.3f} \n".format(tes_f, tes_acc))
#
# print("\n\n ... finished. ")
