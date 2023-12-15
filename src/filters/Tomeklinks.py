import pandas as pd
import numpy as np
import dataclasses
from sklearn.neighbors import NearestNeighbors
from Filter import Filter


@dataclasses.dataclass(frozen=False)
class Tomeklinks(Filter):
    data: pd.DataFrame | None = None
    delete: str = "major"

    def __apply_filter__(self):
        clean_ls =  np.ones(len(self.data), dtype=bool)
        class_count = self.data.iloc[:,-1].value_counts()
        # make sure there are only 2 classes
        if len(class_count) != 2:
            print(
                f"only work on data in 2 classes. found {len(class_count)} classes instead"
            )
            return clean_ls
        major, minor = class_count.index
        # return if balanced data but delete set to "major"
        if class_count[major] == class_count[minor] and self.delete == "major":
            print(
                'balanced data so no cleaning is performed. consider setting delete to "both".'
            )
            return clean_ls
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
            for p1,p2 in links:
                if self.y[p2] == major:
                    clean_ls[p2] = False
                else:
                    clean_ls[p1] = False
        elif self.delete == "both":
            for p1,p2 in links:
                clean_ls[[p1,p2]] = False
        else:
            print('invalid deletion strategy. choose from "major" or "both"')
            return clean_ls
        return clean_ls
