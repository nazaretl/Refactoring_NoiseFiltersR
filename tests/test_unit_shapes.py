from Mode import MODE
from ENN import ENN
from HARF import HARF

from sklearn import datasets
from utils import load_data

def test_shapes():

    df = load_data('Iris')
    filtered_iris = MODE(df)
    noise_index = filtered_iris.find_noise()

    assert filtered_iris.clean_samples().shape[1] == df.shape[1] 
    assert filtered_iris.noisy_samples().shape[1] == df.shape[1] 
    assert filtered_iris.clean_samples().shape[0] + filtered_iris.noisy_samples().shape[0] == df.shape[0] 