"""Test Module for High Agreement Random Forest (HARF)

Runs basic tests for the functionality

"""
import pandas as pd
from utils import get_data, add_random_noise
from filters.HARF import HARF



def test_HARF(noise_level=0.2):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    )
    X, y = get_data(df)
    y_noisy = add_random_noise(y, noise_level=noise_level)

    df_noisy = df.copy()
    df_noisy.iloc[:, -1] = y_noisy
    harf = HARF(df_noisy)
    clean_list = harf.noise_index()
    noisy_ins = harf.noisy_samples()

    noisy = y == y_noisy
    acc = (noisy == clean_list).sum() / len(clean_list)
    print("The HARF filter has accuracy of {}".format(round(acc, 3)))
