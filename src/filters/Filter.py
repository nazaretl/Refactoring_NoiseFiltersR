"""Base class for package filters

Provides common methods and abstract structure to streamline addition of new filters.

"""

import abc
import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=False, eq=False)
class Filter(abc.ABC):
    
    @abc.abstractmethod
    def __apply_filter__(self):
        """
        Special function that applies the filter method on the data

        Returns:
        --------
        List[bool]: 
            A list the length of the input DataFrame of datapoints, with an 
            element being True if the filter determines the corresponding datapoint is signal,
            and False if the datapoint is noise. 
        """
        pass

    def __get_values_and_labels__(self):
        """
        Special function that splits 'data' into two pandas.DataFrame of datapoints and 
        classifications.

        Returns:
        --------
        pandas.DataFrame:
            A DataFrame only containing the datapoints and not the associated classifications

        pandas.DataFrame:
            A DataFrame only containing the classifications and not the associated datapoints
        """
        return self.data.iloc[:, :-1].values, self.data.iloc[:, -1].values

    def remove_noise(self):
        """
        Runs the application of the filter on the input 'data'.

        Returns:
        --------
        List[bool]: 
            A list the length of the input DataFrame of datapoints, with an 
            element being True if the filter determines the corresponding datapoint is signal,
            and False if the datapoint is noise.
        
        """
        self.X, self.y = self.__get_values_and_labels__()
        self.clean_list = self.__apply_filter__()
        return self.clean_list

    def clean_samples(self):
        """
        Returns a list of the data determined as signal by the filter. Runs the filter
        via self.remove_noise() if this has not been done before.

        Returns:
        --------
        List[bool]: 
            A list the length of the input DataFrame of datapoints, with an 
            element being True if the filter determines the corresponding datapoint is signal,
            and False if the datapoint is noise.
        """
        try:
            return self.data[self.clean_list]
        except AttributeError:
            _clean = self.remove_noise
            return self.data[self.clean_list]

    def noisy_samples(self):
        """
        Returns a list of the data determined as noise by the filter. Runs the filter
        via self.remove_noise() if this has not been done before.

        Returns:
        --------
        List[bool]: 
            A list the length of the input DataFrame of datapoints, with an 
            element being True if the filter determines the corresponding datapoint is noise,
            and False if the datapoint is signal.
        """
        try:
            return self.data[np.invert(self.clean_list)]
        except AttributeError:
            _clean = self.remove_noise
            return self.data[np.invert(self.clean_list)]
