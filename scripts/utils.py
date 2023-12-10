import pandas as pd
import random
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.datasets import (
    load_iris,
    load_diabetes,
    load_linnerud,
    load_wine,
    load_breast_cancer,
)


def load_data(dataset):
    sets = {
        "iris": load_iris,
        "diabetes": load_diabetes,
        "linnerud": load_linnerud,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
    }
    dataset = sets[dataset]
    di = dataset()
    data = pd.DataFrame(
        data=np.c_[di["data"], di["target"]], columns=di["feature_names"] + ["target"]
    )
    return data


def get_values_labels(data):
    return data.iloc[:, :-1], data.iloc[:, -1]


def add_random_noise(y, noise_level=0.2):
    labels = list(pd.unique(y))
    n = len(y)
    n_noisy = round(n * noise_level)
    idx_noisy = random.sample(range(n), n_noisy)
    y_noisy = y.copy()
    for idx in idx_noisy:
        clean_label = y[idx]
        labels = list(pd.unique(y))
        labels.remove(clean_label)
        noisy_label = random.choice(labels)
        y_noisy.iloc[idx] = noisy_label

    return y_noisy


def calculate_metrics(y_true, y_pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, fscore
