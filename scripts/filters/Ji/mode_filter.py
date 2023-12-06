import numpy as np
import pandas as pd
from scipy.spatial import distance
import random


def getData(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


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


def mode_filter(df, type="classical", noise_action="repair", beta=1):
    labels = df.iloc[:, -1].unique()

    if type not in ["classical"]:
        raise ValueError("The argument 'type' must be 'classical'")

    # Similarity calculation using Euclidean distance
    similarity = distance.cdist(df.iloc[:, :-1], df.iloc[:, :-1], 'euclidean')

    new_class = df.iloc[:, -1].copy()
    is_clean = np.ones(len(df), dtype=bool)  # Initialize a boolean array

    if type == "classical":
        for i in range(len(df)):
            sums_per_class = []
            for label in labels:
                class_indices = df.iloc[:, -1] == label
                sum_class = np.sum(similarity[i, class_indices])
                sum_other = np.sum(np.exp(-beta) * similarity[i, ~class_indices])
                sums_per_class.append(sum_class)

            # Assign the label with the highest sum as the new class label
            new_label = labels[np.argmin(sums_per_class)]
            new_class.iloc[i] = new_label

            # Check if the label has changed (misclassified)
            is_clean[i] = new_label == df.iloc[i, -1]

    if noise_action == "repair":
        # If noise_action is "repair," create a copy of the DataFrame and replace the last column
        df_clean = df.copy()
        df_clean.iloc[:, -1] = new_class.values  # Assign values to ensure alignment
    else:
        # If noise_action is not "repair," filter the DataFrame to keep only rows with corrected labels
        df_clean = df[new_class == df.iloc[:, -1]].copy()

    return is_clean



# Example usage
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

X, y = getData(df)
y_noisy = add_random_noise(y, noise_level=0.2)

df_noisy = df.copy()
df_noisy.iloc[:, -1] = y_noisy

clean_list = mode_filter(df_noisy, type="classical", noise_action="repair", beta=1)

noisy = y == y_noisy
print(clean_list)

acc = (noisy == clean_list).sum() / len(clean_list)
print('The Mode Filter has accuracy of {}'.format(round(acc, 3)))
