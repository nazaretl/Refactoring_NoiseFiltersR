import pandas as pd
import numpy as np
import dataclasses
from sklearn.neighbors import KNeighborsClassifier

import sys
sys.path.append('.')
from Filter import Filter

@dataclasses.dataclass(frozen=False, eq=False, init=False)

class ENN(Filter):

   # X: FArray
   # y: FArray
  #  data: pd.core.frame.DataFrame
   # n_neighbors: int

    def __init__(self, data, n_neighbors = 5):
        self.data = data # dataframe that preserves the feature names
        self.X, self.y = self.__get_values_and_labels__()
        self.n_neighbors = n_neighbors

  #  def __post_init__(self):
      #  object.__setattr__(self, 'n_neighbors',  n_neighbors)
#        object.__setattr__(self, 'data',  data)
   #     object.__setattr__(self, 'X',  data.iloc[:,:-1].values)
    #    object.__setattr__(self, 'y',  data.iloc[:,-1].values)

    
    def __apply_filter__(self):

        N = len(self.data)
        clean_list = []

        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(self.X, self.y)
        for i in range(N):
            own_label = self.y[i]
            neighbors_idx = knn.kneighbors(self.X[i,:].reshape(1, -1), return_distance=False)
            neighbor_labels = self.y[neighbors_idx][0]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            most_frequent_label = values[counts == counts.max()]
            clean_list.append([most_frequent_label == own_label][0][0])
        return clean_list

    def noise_index(self):
        self.clean_list = self.__apply_filter__()
        return self.clean_list
        
