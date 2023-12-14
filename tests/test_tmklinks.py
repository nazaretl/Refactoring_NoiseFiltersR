import pandas as pd
import numpy as np
from filters.Tomeklinks import Tomeklinks


def test1():
    np.random.seed(0)
    df1 = pd.DataFrame(np.random.randint(0, 70, size=(8, 2)))
    df1["label"] = [0 for _ in range(len(df1))]
    df2 = pd.DataFrame(np.random.randint(40, 100, size=(15, 2)))
    df2["label"] = [1 for _ in range(len(df2))]
    dftot = pd.concat([df1, df2], ignore_index=True)
    tmkfilter = Tomeklinks(dftot,delete='major')
    ind = tmkfilter.noise_index()
    noise = tmkfilter.noisy_samples()
    assert list(noise.index.values) == [10,17]