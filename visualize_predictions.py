import os
import sys

import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import seaborn as sns


class SCVisualise(object):
    """ 
    """   
    def get_features(self, path):
        dataframe = pd.read_csv(path, sep="\t", header=None)
        return dataframe.values.tolist()

    def plot_TSNE(self, features):
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(features, )
        sns.scatterplot(data=projections)
        plt.grid(True)
        plt.show()

    def plot_UMAP(self, features):
        umap_2d = UMAP(n_components=2, init='random', random_state=0)
        proj_2d = umap_2d.fit_transform(features)
        sns.scatterplot(data=proj_2d)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    scVis = SCVisualise()

    # filter single-cell data
    data = scVis.get_features("data/output.csv")
    
    scVis.plot_TSNE(data)
    
    scVis.plot_UMAP(data)
