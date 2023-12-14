import pandas as pd
import numpy as np
import dataclasses
from sklearn.neighbors import NearestNeighbors
from Filter import Filter

@dataclasses.dataclass(frozen=False, eq=False, init=False)
class Tomeklinks(Filter):

    def __init__(self, df, delete="major"):
        self.data = df
        self.X, self.y = self.__get_values_and_labels__()
        self.delete = delete
    
    def __apply_filter__(self):
        clean_ls = np.arange(len(self.data))
        class_count = self.data.iloc[:,-1].value_counts()
        # make sure there are only 2 classes
        if len(class_count) != 2:
            print(
                f"only work on data in 2 classes. found {len(class_count)} classes instead"
            )
            return clean_ls,[]
        major, minor = class_count.index
        # return if balanced data but delete set to "major"
        if class_count[major] == class_count[minor] and delete == "major":
            print(
                'balanced data so no cleaning is performed. consider setting delete to "both".'
            )
            return clean_ls,[]
        # find tomeklinks
        nns = NearestNeighbors(n_neighbors=1).fit(self.X, self.y).kneighbors()[1]
        dictnn = {}
        links = []
        for i in range(len(nns)):
            nn = nns[i][0]
            if self.y[i] == self.y[nn]:
                continue
            if i in dictnn and nn == dictnn[i]:
                links.append((i, nn))
            dictnn[nn] = i
        # drop tomeklinks
        if self.delete == "major":
            for i in range(len(links)):
                p1,p2 = links[i]
                if self.y[p2] == major:
                    links[i] = (p2,p1)
            clean_ls = np.delete(clean_ls,[pair[0] for pair in links])
        elif self.delete == "both":
            clean_ls = np.delete(clean_ls, sum(links,()))
        else:
            print('invalid deletion strategy. choose from "major" or "both"')
            return clean_ls,links
        return clean_ls,links

    def noise_index(self):
        self.clean_list, self.links = self.__apply_filter__()
        return self.clean_list

    def noisy_samples(self):
        return self.data.iloc[list(sum(self.links,()))]