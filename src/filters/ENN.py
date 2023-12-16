"""Enhanced Nearest Neighbors (ENN) Filter

Similarity-based filter for removing label noise 
from a dataset as a preprocessing step of classification.

"""

import numpy as np
import dataclasses
from sklearn.neighbors import KNeighborsClassifier
from Filter import Filter
import pandas as pd


@dataclasses.dataclass(frozen=False)
class ENN(Filter):
    """
    Ensemble-based filter for removing label noise from a dataset as a
    preprocessing step of classification. Does not reclassify data.

    Attributes
    ----------

    data: pandas.DataFrame
        The two-dimensional numerical data to be filtered, organized such that each column is
        a property and each row is a datapoint. The last column should contain the classification of
        each datapoint.

    n_neigbors: int, default=5
        The number of neighbor groups to assign to input data in classification.
    """
    data: pd.DataFrame | None = None
    n_neighbors: int = 5

    def __apply_filter__(self):
        N = len(self.data)
        clean_list = []

        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(self.X, self.y)
        for i in range(N):
            own_label = self.y[i]
            neighbors_idx = knn.kneighbors(
                self.X[i, :].reshape(1, -1), return_distance=False
            )
            neighbor_labels = self.y[neighbors_idx][0]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            most_frequent_label = values[counts == counts.max()]
            clean_list.append([most_frequent_label == own_label][0][0])
        return clean_list
