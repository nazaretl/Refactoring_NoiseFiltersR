import abc
import dataclasses
import pandas as pd
import numpy as np

@dataclasses.dataclass(frozen=False, eq=False)
class Filter(abc.ABC):
    @abc.abstractmethod
    def __apply_filter__(self):
        pass

    @abc.abstractmethod
    def noisy_samples(self):
        pass

    @abc.abstractmethod
    def clean_samples(self):
        pass

    def clean_samples(self):
        return self.data[self.data[clean_list]]

    def noisy_samples(self):
        return self.data[np.invert(self.clean_list)]

    def __get_values_and_labels__(self):
        return self.data.iloc[:,:-1].values, self.data.iloc[:,-1].values 
