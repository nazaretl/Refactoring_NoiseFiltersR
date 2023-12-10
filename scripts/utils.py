import pandas as pd
import random 

def get_data(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X, y

def add_random_noise(y, noise_level = 0.2):

        labels = list(pd.unique(y))
        n = len(y)
        n_noisy = round(n * noise_level)
        idx_noisy = random.sample(range(n), n_noisy)
        y_noisy = y.copy()
        for idx in idx_noisy:
            clean_label = y[idx]
            labels = list(pd.unique(y))
            labels.remove(clean_label)
            noisy_label = random.choice(labels)
            y_noisy.iloc[idx] = noisy_label
    
        return y_noisy