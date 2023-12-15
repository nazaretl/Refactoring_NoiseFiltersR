from Mode import MODE
from ENN import ENN
from HARF import HARF
from Tomeklinks import Tomeklinks

from sklearn import datasets
from utils import load_data
import pytest

@pytest.mark.parametrize("filter", (ENN, MODE, HARF, Tomeklinks))
def test_shapes(filter):

    df = load_data('Iris')
    filtered_iris = filter(df)
    noise_index = filtered_iris.remove_noise()

    assert filtered_iris.clean_samples().shape[1] == df.shape[1] 
    assert filtered_iris.noisy_samples().shape[1] == df.shape[1] 
    assert filtered_iris.clean_samples().shape[0] + filtered_iris.noisy_samples().shape[0] == df.shape[0] 