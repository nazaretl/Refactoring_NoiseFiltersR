import abc
import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=False, eq=False)
class Filter(abc.ABC):
    @abc.abstractmethod
    def __apply_filter__(self):
        pass

    def __get_values_and_labels__(self):
        return self.data.iloc[:, :-1].values, self.data.iloc[:, -1].values

    def remove_noise(self):
        self.X, self.y = self.__get_values_and_labels__()
        self.clean_list = self.__apply_filter__()
        return self.clean_list

    def clean_samples(self):
        return self.data[self.clean_list]

    def noisy_samples(self):
        return self.data[np.invert(self.clean_list)]
