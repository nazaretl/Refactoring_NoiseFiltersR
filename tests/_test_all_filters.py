import pandas as pd

from utils import get_data, add_random_noise, calculate_metrics
from ENN import ENN
from Mode import MODE
from HARF import HARF

import pytest

from timeit import default_timer as timer


noise_level = 0.2
df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)
X, y = get_data(df)
y_noisy = add_random_noise(y, noise_level=noise_level)

df_noisy = df.copy()
df_noisy.iloc[:, -1] = y_noisy


def pytest_generate_tests(metafunc):
    if "filter" in metafunc.fixturenames:
        list_filters = [ENN(df_noisy), MODE(df_noisy), HARF(df_noisy)]
        metafunc.parametrize("filter", list_filters)


def test_Iris(filter):
    start = timer()
    clean_list = filter.noise_index()
    end = timer()
    noisy_ins = filter.noisy_samples()

    noisy = y == y_noisy
    accuracy, precision, recall, fscore = calculate_metrics(noisy, clean_list)
    to_save = pd.DataFrame(
        [accuracy, precision, recall, fscore, end - start, noise_level]
    ).T.round(4)
    to_save.index = [filter.__class__.__name__]
    print(to_save)
    to_save.to_csv("output/results.csv", mode="a", sep="\t", index=True, header=False)


# @pytest.mark.parametrize("filter", (ENN(df_noisy)))
# def test_Iris(filter):
