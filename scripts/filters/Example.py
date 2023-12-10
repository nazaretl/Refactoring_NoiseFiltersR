import pandas as pd
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random

# from sklearn import datasets
# iris = datasets.load_iris()


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


# original filter https://rdrr.io/cran/NoiseFiltersR/src/R/ENN.R
def ENN(df, n_neighbors=5):
    """
    A filter that finds n neibhours for each datapoint and drops the datapoint (FALSE in the clean_list)
    if the majority of the neibhors has a different label
    Returns a boolean list with TRUE for clean and FALSE for noisy data point
    """
    X, y = getData(df)
    N = len(df)
    clean_list = []

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X.values, y.values)
    for i in range(N):
        own_label = y[i]
        neighbors_idx = knn.kneighbors(
            X.iloc[i, :].values.reshape(1, -1), return_distance=False
        )
        neighbor_labels = y.values[neighbors_idx][0]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        most_frequent_label = values[counts == counts.max()]
        clean_list.append([most_frequent_label == own_label][0][0])

    # clean_samples =  df[clean_list]
    # noisy_samples = df[np.invert(clean_list)]
    return clean_list


# Load the iris dataset into a Pandas DataFrame
df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

X, y = getData(df)
y_noisy = add_random_noise(y, noise_level=0.2)

df_noisy = df.copy()
df_noisy.iloc[:, -1] = y_noisy

clean_list = ENN(df_noisy, n_neighbors=5)

noisy = y == y_noisy
acc = (noisy == clean_list).sum() / len(clean_list)
print(
    "The ENN filter has accuracy of {} on the Iris dataset with 20% of noise.".format(
        round(acc, 3)
    )
)
