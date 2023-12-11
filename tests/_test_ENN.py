import pandas as pd
from utils import get_data, add_random_noise
from filters.ENN import ENN


def test_ENN(noise_level=0.2):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    )
    X, y = get_data(df)
    y_noisy = add_random_noise(y, noise_level=noise_level)

    df_noisy = df.copy()
    df_noisy.iloc[:, -1] = y_noisy
    enn = ENN(df_noisy)
    clean_list = enn.noise_index()
    noisy_ins = enn.noisy_samples()

    noisy = y == y_noisy
    acc = (noisy == clean_list).sum() / len(clean_list)
    print("The ENN filter has accuracy of {}".format(round(acc, 3)))
