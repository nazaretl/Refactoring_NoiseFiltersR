"""Test Module for High Agreement Random Forest (HARF)

Runs basic tests for the functionality

"""

from HARF import HARF
from sklearn import datasets

def test_HARF():
    iris = datasets.load_iris(as_frame=True)

    filtered_iris = HARF(iris.frame, nfolds=12, ntrees=150, agreementLevel=0.71)
    assert filtered_iris.clean_samples().shape[1] == 5
    assert filtered_iris.noisy_samples().shape[1] == 5
    assert filtered_iris.clean_samples().shape[0] + filtered_iris.noisy_samples().shape[0] == iris.frame.shape[0]