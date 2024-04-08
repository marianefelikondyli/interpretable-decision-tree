import numpy as np

""" ML model performance evaluation metrics """


def accuracy(preds, labels):
    """Calculates accuracy given two numpy arrays"""
    return np.mean(preds == labels)


def confusion_matrix(preds, labels, num_of_labels):
    """Creates a confusion matrix from given two numpy arrays"""
    unique_label_classes = np.sort(np.unique(labels))
    int_encoded_labels = np.searchsorted(unique_label_classes, labels)
    int_encoded_pred = np.searchsorted(unique_label_classes, preds)

    matrix = np.zeros((num_of_labels, num_of_labels))
    for i in range(len(preds)):
        matrix[int_encoded_pred[i], int_encoded_labels[i]] += 1
    return matrix


def precision_and_recall(preds, labels):
    """Returns individual Precision and Recall values of each class"""
    num_of_labels = len(np.unique(labels))
    matrix = confusion_matrix(preds, labels, num_of_labels)
    r = []
    p = []

    number_of_not_NA_precision, number_of_not_NA_recall, precision_sum, recall_sum = (
        0,
        0,
        0,
        0,
    )

    for i in range(num_of_labels):
        TP = float(matrix[i, i])
        TP_FP = np.sum(matrix[i, :])
        TP_FN = np.sum(matrix[:, i])

        if TP_FN != 0:
            recall_sum += TP / TP_FN
            number_of_not_NA_recall += 1

        if TP_FP != 0:
            precision_sum += TP / TP_FP
            number_of_not_NA_precision += 1

    recall = (
        recall_sum / number_of_not_NA_recall if number_of_not_NA_recall != 0 else "NA"
    )
    precision = (
        precision_sum / number_of_not_NA_precision
        if number_of_not_NA_precision != 0
        else "NA"
    )

    return (precision, recall)


def fscore(preds, labels):
    """Calculates macro f score given two numpy arrays"""
    pred_labels, pred_numberized = np.unique(preds, return_inverse=True)
    labels, labels_numberized = np.unique(labels, return_inverse=True)
    p, r = precision_and_recall(pred_numberized, labels_numberized)
    if p == "NA" or r == "NA" or p + r == 0:
        return "NA"

    return 2 * p * r / (p + r)