import pandas as pd
import numpy as np

from utils import get_data, add_random_noise
from Mode import MODE


def test_MODE(noise_level=0.2):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    )
    X, y = get_data(df)
    y_noisy = add_random_noise(y, noise_level=noise_level)

    df_noisy = df.copy()
    df_noisy.iloc[:, -1] = y_noisy
    Modefilter = MODE(df_noisy, 1)
    clean_list = Modefilter.noise_index()
    celan = Modefilter.clean_samples()
    print(celan)
    nn = Modefilter.noisy_samples()
    print(nn)
    noisy = y == y_noisy
    acc = (noisy == clean_list).sum() / len(clean_list)
    print("The MODE filter has accuracy of {}".format(round(acc, 3)))
