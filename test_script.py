import time
import copy

import pandas as pd
from sklearn.model_selection import KFold

from decision_tree import Decision_Tree, print_tree, Leaf


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


""" Error Analysis for Classification """

kf = KFold(n_splits=5, random_state=24, shuffle=True)
print(""" \n---------------------- PART 1 ---------------------------------\n\n\n""")

scores_array = []
for max_depth in [2]:
    row = []
    print("max_depth : " + str(max_depth))
    i = 1

    test_F_scores, test_accuracies, test_ps, test_rs = [], [], [], []
    train_F_scores, train_accuracies, train_ps, train_rs = [], [], [], []

    start = time.time()
    for train_index, test_index in kf.split(X):  # Each FOLD
        print("   fold" + str(i) + " :    ", end=" ")
        i += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model = Decision_Tree(max_depth, header)
        model.train(X_train, y_train, mode_of_each_column)
        print_tree(model.root)
        train_acc, train_f, train_p, train_r = model.test(X_train, y_train)
        print(
            "Train : F1 Score: {:.3f}, Accuracy: {:.3f}     ".format(
                train_f, train_acc
            ),
            end="",
        )

        test_acc, test_f, test_p, test_r = model.test(X_test, y_test)

        test_accuracies.append(test_acc)
        test_F_scores.append(test_f)
        test_ps.append(test_p)
        test_rs.append(test_r)

        train_accuracies.append(train_acc)
        train_F_scores.append(train_f)
        train_ps.append(train_p)
        train_rs.append(train_r)
        print(
            " TEST : F1 Score: {:.3f} ,  Accuracy: {:.3f}  , Precision: {:.3f} , Recall: {:.3f}".format(
                test_f, test_acc, test_p, test_r
            )
        )



    print(
        "   AVERAGE :                                                        F1 Score: {:.3f} ,  Accuracy: {:.3f}  , Precision: {:.3f} , Recall: {:.3f}".format(
            sum(test_F_scores) / 5,
            sum(test_accuracies) / 5,
            sum(test_ps) / 5,
            sum(test_rs) / 5,
        )
    )

    row.extend(
        [
            max_depth,
            sum(train_F_scores) / 5,
            sum(test_F_scores) / 5,
            sum(train_accuracies) / 5,
            sum(test_accuracies) / 5,
            sum(train_ps) / 5,
            sum(test_ps) / 5,
            sum(train_rs) / 5,
            sum(test_rs) / 5,
        ]
    )
    scores_array.append(row)
    finish = time.time()
    seconds = finish - start
    minutes = seconds // 60
    seconds -= 60 * minutes
    print("Elapsed time: %d:%d   minutes:seconds \n" % (minutes, seconds))

scores_df = pd.DataFrame(
    scores_array,
    columns=[
        "max_depth",
        "train_f1",
        "test_f1",
        "train_accuracy",
        "test_accuracy",
        "train_precision",
        "test_precision",
        "train_recall",
        "test_recall",
    ],
)
print("\nAverage of 5 folds:\n")
print(scores_df)

#
# """ Print Best Performed Tree """
#
# kf = KFold(n_splits=5, random_state=24, shuffle=True)
# this_fold = 0
# for train_index, test_index in kf.split(X):
#     # we only need fold_4
#     this_fold += 1
#     if this_fold != 4:
#         continue
#
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = Y[train_index], Y[test_index]
#
#     model = Decision_Tree(maximum_depth=2, header=header)
#     model.train(X_train, y_train, mode_of_each_column)
#
#     train_acc, train_f, train_p, train_r = model.test(X_train, y_train)
#
#     test_acc, test_f, test_p, test_r = model.test(X_test, y_test)
#     break
#
# print("\n\n\n BEST PERFORMING TREE  max_depth=2 : \n")
# print_tree(model.root)
