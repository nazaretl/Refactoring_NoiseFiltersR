import pandas as pd
import numpy as np
from scripts.filters.Mode import MODE  

def get_Data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def add_random_noise(y, noise_level=0.2):
    """
    Function to add random noise to the labels.
    """
    labels = list(pd.unique(y))
    n = len(y)
    n_noisy = round(n * noise_level)
    idx_noisy = np.random.choice(range(n), n_noisy, replace=False)
    y_noisy = y.copy()
    for idx in idx_noisy:
        clean_label = y[idx]
        labels = list(pd.unique(y))
        labels.remove(clean_label)
        noisy_label = np.random.choice(labels)
        y_noisy.iloc[idx] = noisy_label
    return y_noisy

def test_MODE(noise_level=0.2):
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    X, y = get_Data(df)
    y_noisy = add_random_noise(y,noise_level = noise_level)
    
    df_noisy = df.copy()
    df_noisy.iloc[:,-1] = y_noisy
    Modefilter = MODE(df_noisy,1)
    clean_list = Modefilter.noise_index()
    celan=Modefilter.clean_samples()
    print(celan)
    nn=Modefilter.noisy_samples()
    print(nn)
    noisy = y == y_noisy
    acc = (noisy == clean_list).sum()/len(clean_list)
    print('The MODE filter has accuracy of {}'.format(round(acc,3)))
    
test_MODE()


