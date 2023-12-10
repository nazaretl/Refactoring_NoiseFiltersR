import pandas as pd
import numpy as np
import dataclasses
from scipy.spatial import distance

from Filter import Filter  

@dataclasses.dataclass(frozen=False, eq=False, init=False)
class MODE(Filter):
    def __init__(self, df, beta):
        self.df = df
        self.X, self.y = self.__get_values_and_labels__(self.df)
        self.beta = beta

    def __apply_filter__(self):
        is_clean=[]
        labels = self.df.iloc[:, -1].unique()
        similarity = distance.cdist(self.X, self.X, 'euclidean')
        new_class = self.df.iloc[:, -1].copy()
        is_clean = np.ones(len(self.df), dtype=bool)

        for i in range(len(self.df)):
            sums_per_class = []
            for label in labels:
                class_indices = self.df.iloc[:, -1] == label
                sum_class = np.sum(similarity[i, class_indices])
                sums_per_class.append(sum_class)

            new_label = labels[np.argmin(sums_per_class)]
            new_class.iloc[i] = new_label
            is_clean[i] = new_label == self.df.iloc[i, -1]


        return is_clean

    def noise_index(self):
        self.clean_list = self.__apply_filter__()
        return self.clean_list
        
    def clean_samples(self):
        return self.df[self.clean_list]

    def noisy_samples(self):
        return self.df[~self.clean_list]