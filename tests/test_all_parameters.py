import pandas as pd

from utils import load_data, get_values_labels, add_random_noise, calculate_metrics
from ENN import ENN
from Mode import MODE
from HARF import HARF
from Tomeklinks import Tomeklinks

import pytest

from timeit import default_timer as timer


@pytest.mark.parametrize("noise_level", (0.1, 0.2))
@pytest.mark.parametrize(
    "dataset",
    ("Iris", "Diabetes", "Wine", "Magic", "DryBean"),
)
@pytest.mark.parametrize("filter", (ENN, MODE, HARF, Tomeklinks))
def test_all_parameters(noise_level, dataset, filter):
    df = load_data(dataset)
    X, y = get_values_labels(df)
    y_noisy = add_random_noise(y, noise_level=noise_level)

    df_noisy = df.copy()
    df_noisy.iloc[:, -1] = y_noisy

    start = timer()
    filter = filter(df_noisy)
    clean_list = filter.remove_noise()
    end = timer()
    noisy_ins = filter.noisy_samples()

    noisy = y == y_noisy
    accuracy, precision, recall, fscore = calculate_metrics(noisy, clean_list)
    to_save = pd.DataFrame(
        [accuracy, precision, recall, fscore, end - start, noise_level]
    ).T.round(4)
    to_save.index = [filter.__class__.__name__]
    to_save["dataset"] = dataset

    to_save.to_csv("output/results.csv", mode="a", sep="\t", index=True, header=False)
