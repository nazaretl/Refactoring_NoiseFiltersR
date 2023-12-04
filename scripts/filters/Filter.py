import abc
import dataclasses
import pandas as pd

@dataclasses.dataclass(frozen=True, eq=False)
class Filter(abc.ABC):
    @abc.abstractmethod
    def __apply_filter__(self, data):
        pass

    @abc.abstractmethod
    def noisy_samples(self, data):
        pass

    @abc.abstractmethod
    def clean_samples(self, data):
        pass

    def __get_values_and_labels__(self, data):
        self.X = data.iloc[:,:-1].values
        self.y = data.iloc[:,-1].values
        return self.X, self.y 
