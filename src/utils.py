import pandas as pd
import random
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.datasets import (
    load_iris,
    load_diabetes,
    load_wine
)


def load_data(dataset):

    """
    Loads the data into a pandas dataframe

    Args:
      dataset (str): name of the dataset

    Returns:
      pandas.DataFrmae: pandas dataframe where the last
                        column contains the labels
    
    """
    
    sets = {
        "Iris": load_iris,
        "Diabetes": load_diabetes,
        "Wine": load_wine,
    }
    if dataset in list(sets.keys()):
        dataset = sets[dataset]
        di = dataset()
        data = pd.DataFrame(
            data=np.c_[di["data"], di["target"]],
            columns=di["feature_names"] + ["target"],
        )
    else:
        # for datasets Magic, Adult and DryBean
        data = pd.read_csv(
            "datasets/" + dataset + ".csv.gz", sep="\t", compression="zip"
        )
    return data


def get_values_labels(data):
    """
    Returns the values and labels
    from a dataframe that contains both

    Args:
      data (pandas.DataFrame): dataframe containing values and labels

    Returns:
      pandas.DataFrame: dataframe containing only values
      pandas.Series: series containing only labels
    
    """
    return data.iloc[:, :-1], data.iloc[:, -1]


def add_random_noise(y, noise_level=0.2):

    """
    Adds random noise to labels

    Args:
      y (pandas.Series): clean labels

    Returns:
      pandas.Series: noisy labels
    
    """
    
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

    """
    Calculates performance metrics for filters

    Args:
      y_true (pandas.Series|list): list of true labels
      y_pred (pandas.Series|list): list of predicted labels


    Returns:
      float: accuracy score
      float: precision score
      float: recall score
      float: F-score 

    
    """
    
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, fscore

