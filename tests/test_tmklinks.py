# import sys
# sys.path.insert(0, 'C:/Users/jihci/OneDrive/Desktop/apc524/project/scripts')

# from scripts.filters import tomeklinks

#import scripts
import pandas as pd
import numpy as np
from filters import tomeklinks
def test1():
    np.random.seed(0)
    df1 = pd.DataFrame(np.random.randint(0,70,size=(8, 2)))
    df1['label'] = [0 for _ in range(len(df1))]
    df2 = pd.DataFrame(np.random.randint(40,100,size=(15, 2)))
    df2['label'] = [1 for _ in range(len(df2))]
    dftot = pd.concat([df1,df2], ignore_index=True)
    dfcleaned,links = tomeklinks.tomeklinks(dftot)
    assert links == [(10, 0), (17, 5)]