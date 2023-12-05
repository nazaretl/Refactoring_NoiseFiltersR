"""High Agreement Random Forest (HARF)

Ensemble-based filter for removing label noise from a dataset as a
preprocessing step of classification.




"""
import dataclasses
import random as rand
import itertools
import numpy.typing as nptypes
import numpy as np
    

@dataclasses.dataclass(frozen=True)
class HARF():
    """
    TODO: Formalize docstrings and explanations

    Parameters
    ----------
    x : numpy.types.ArrayLike
        The two-dimensional numerical data to be filtered, organized such that each column is 
        a property and each row is a datapoint.

    classColumn: numpy.types.ArrayLike
        A one-dimensional array-like of classifications for the datapoints numerically described in 
        the rows of x

    
    """
    x: nptypes.ArrayLike = None
    classes: nptypes.ArrayLike = None
    # formula: 
    # TODO: Add compatibility for formula inputs
    nfolds: int = 10
    agreementLevel: float = 0.7
    ntrees: int = 500

    x = np.array(x, copy=True, ndmin=2)
    classes = np.array(classes, copy=True, ndmin=1)

    if x.ndim != 2:
        raise AttributeError("Data must be two-dimensional")
    if(agreementLevel<0.5 | agreementLevel>1):
        raise ValueError("The agreement level must range between 0.5 and 1")
    if classes.shape[0] != x.shape[1]:
        raise ValueError("classColumn and number of rows in x must match")
    if classes.dtype != str:
        raise ValueError("classColumn must be an array-like of strings")
    
    # Split indices of the data into randomized chunks
    folds = itertools.batched(rand.shuffle(range(classes.size)), nfolds)
                            
    # Create array to hold decision of filtering method
    isNoise = np.full(classes.shape, False, dtype=bool)
    
   
    







    
