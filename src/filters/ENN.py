import numpy as np
import dataclasses
from sklearn.neighbors import KNeighborsClassifier
from Filter import Filter
import pandas as pd


@dataclasses.dataclass(frozen=False)
class ENN(Filter):

    """
       Applies the Edited Nearest Neighbors (ENN) filter. 
       Based on the ENN filter from NoiseFiltersR 
       (https://rdrr.io/cran/NoiseFiltersR/man/ENN.html) 
    
        Args:
          df (pandas.DataFrame): a pandas dataframe. 
                        The last column must contain noisy labels
          n_neighbors (int): number of neighbors to be considered
          
        Returns:
          list: a boolean list with True if instance is consedered
                  to be clean, False otherwise
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
