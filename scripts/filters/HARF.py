"""High Agreement Random Forest (HARF)

Ensemble-based filter for removing label noise from a dataset as a
preprocessing step of classification. Does not reclassify data.

NOTE: Class names in classes get truncated to their first character, can cause issues

TODO: Formalize docstrings

"""
from FilteredData import FilteredData
import numpy.typing as nptypes
import numpy as np
import sklearn.ensemble as skensemble

def HARF(x: nptypes.ArrayLike | None = None, 
        classes: nptypes.ArrayLike | None = None, nfolds: int = 10, 
        agreementLevel: float = 0.7, ntrees: int = 500) -> FilteredData:
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
    
    # TODO: Add compatibility for formula inputs
    
    
    if x.ndim != 2:
        raise AttributeError(f"Data must be two-dimensional, currently {x.ndim}-dimensional")
    if (agreementLevel<0.5) | (agreementLevel>1):
        raise ValueError(f"The agreement level of {agreementLevel} must be between 0.5 and 1")
    if classes.shape[0] != x.shape[0]:
        raise ValueError(f"The number of rows in x ({x.shape[0]}) and in classes ({classes.shape[0]}) must match")
    if str(classes.dtype) not in "<U1S":
        raise ValueError(f"classes must be an array-like of Unicode or strings, currently {classes.dtype}")
    
    x = np.array(x, copy=True, ndmin=2)
    classes = np.array(classes, copy=True, ndmin=1)

    rand_forest = skensemble.RandomForestClassifier(n_estimators=ntrees, max_samples=nfolds, n_jobs=-1)
    
    rand_forest.fit(x, classes)
    prob = rand_forest.predict_proba(x)

    # Find the elements which do not fit within the agreement limit
    is_noise = (prob<agreementLevel).all(axis=1)
    
    # Build Filter class
    clean_data = x[np.logical_not(is_noise)]
    removed_idx = np.nonzero(is_noise)
    repaired_idx = None
    repaired_labels = None
    parameters = {"nfolds" : nfolds,
                  "agreementLevel" : agreementLevel,
                  "ntrees" : ntrees,
                  }
    call = (f"HARF(x = [{type(x)}, shape={x.shape}] , classes = [{type(classes)}, shape={classes.shape}], "  
            f"nfolds {nfolds}, agreementLevel = {agreementLevel}, ntrees = {ntrees})")
    extra_info = None

    return FilteredData(clean_data, removed_idx, repaired_idx, repaired_labels, parameters, call, extra_info)
    



    
