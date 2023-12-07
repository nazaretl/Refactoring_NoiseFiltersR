"""Test Module for High Agreement Random Forest (HARF)

Runs basic tests for the functionality

"""

from HARF import HARF
from FilteredData import FilteredData
from sklearn import datasets
import numpy as np

def test_HARF():
    iris = datasets.load_iris()

    iris_targets = np.full(len(iris.data[:,0]), "")
    for i in range(len(iris.data[:,0])):
        # Classifcation names get truncated 
        if iris.target[i] == 0:
            iris_targets[i] = "s" # Setosa
        elif iris.target[i] == 1:
            iris_targets[i] = "e" # vErsicolor
        else:
            iris_targets[i] = "i" # vIrginica


    
    filtered_iris = HARF(iris.data, np.transpose(iris_targets), nfolds=12, ntrees=150, agreementLevel=0.71)
    assert filtered_iris.clean_data.shape[1] == 4
    assert filtered_iris.repaired_idx is None
    assert filtered_iris.repaired_labels is None
    assert filtered_iris.parameters["ntrees"] == 150
    assert isinstance(filtered_iris.call, str)
    assert filtered_iris.extra_info is None
