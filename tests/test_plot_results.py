import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


import pytest

def test_all_parameters():
    df = pd.read_csv('output/results.csv', sep = "\t", header = None)
    df.columns = ["Filter","Accuracy", "Precision", "Recall", "Fscore", "Time","NoiseLevel", "Dataset"]
    
    fig, axs = plt.subplots(2,2,figsize=(16,10))
    scores = ["Accuracy", "Precision", "Recall", "Time"]
    for i, score in enumerate(scores):
        ax = axs.flatten()[i] 
        sns.boxplot(data=df, x="Filter", y=score, whis=(0, 100), ax = ax, hue = "Filter")
    name = 'Comparison'
    plt.savefig('figures/'+name+'.png',transparent=False,bbox_inches = 'tight', dpi = 150)    
