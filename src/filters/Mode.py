"""Mode Filter
Similarity-based filter for removing or repairing label noise from a dataset as 
a preprocessing step of classification.

"""
import numpy as np
import dataclasses
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Filter import Filter
import pandas as pd


@dataclasses.dataclass(frozen=False)
class MODE(Filter):
    """
    Attributes
    ----------
    data : pandas.DataFrame
        The two-dimensional numerical data to be filtered, organized such that each column is
        a property and each row is a datapoint. The last column should contain the classification of
        each datapoint.

    """

    data: pd.DataFrame | None = None

    def __calculate_beta__(self):
        # Beta is calculated using variance of the scaled features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        return np.var(X_scaled) * 0.1

    def __apply_filter__(self):
        # Automatically adjust beta based on data
        # Better performance expected with relevant training data
        self.beta = self.__calculate_beta__()

        # make code faster
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        pca = PCA(n_components=min(5, len(self.X[0])))
        X_pca = pca.fit_transform(X_scaled)

        labels = self.data.iloc[:, -1].unique()
        similarity = distance.cdist(X_pca, X_pca, "cosine")
        is_clean = np.zeros(len(self.data), dtype=bool)

        # dynamic beta
        # k = max(5, len(self.X) // 20)
        # local_density = np.sort(similarity[i, :])[k]
        # normalized_density = (local_density - np.min(similarity)) / (np.max(similarity) - np.min(similarity))
        # dynamic_beta = min(self.beta * normalized_density, 10)

        # Precompute
        sums_by_label = {
            label: np.sum(similarity[:, self.data.iloc[:, -1] == label], axis=1)
            for label in labels
        }

        for i in range(len(self.data)):
            sums_per_class = [
                sum_class[i]
                + np.exp(-self.beta) * (np.sum(similarity[i]) - sum_class[i])
                for sum_class in sums_by_label.values()
            ]
            new_label = labels[np.argmin(sums_per_class)]
            is_clean[i] = new_label == self.data.iloc[i, -1]

        return is_clean
