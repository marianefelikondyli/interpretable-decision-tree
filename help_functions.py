import numpy as np
import pandas as pd

""" Helper functions that are used at ID3 Decision Tree Algorithm"""


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


def class_counts(numpy_arr):
    """Returns the unique elements that are inside the
    numpy array, and their individual counts"""

    if len(numpy_arr.shape) == 2:
        valueCounts = pd.Series(numpy_arr[:, 0]).value_counts()
        return np.array(valueCounts.keys().to_list()), np.array(valueCounts.to_list())
    else:
        valueCounts = pd.Series(numpy_arr).value_counts()
        return np.array(valueCounts.keys().to_list()), np.array(valueCounts.to_list())