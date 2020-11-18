import os
import sys

import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
import scanpy as sc
import scipy.sparse as sp_sparse

from matplotlib import pyplot as plt
import seaborn as sns


class SCVisualise(object):
    """ 
    """   
    def get_features(self, path):
        print("Loading dataframe...")
        dataframe = pd.read_csv(path, sep="\t", header=None)
        return dataframe.values.tolist()

    def plot_TSNE(self, features):
        tsne = TSNE(n_components=2, random_state=0)
        print("Computing projections...")
        projections = tsne.fit_transform(features, )
        print("Plotting...")
        sns.scatterplot(data=projections)
        plt.grid(True)
        plt.show()

    def plot_UMAP(self, features):
        umap_2d = UMAP(n_components=2, init='random', random_state=0)
        print("Computing projections...")
        proj_2d = umap_2d.fit_transform(features)
        print("Plotting...")
        sns.scatterplot(data=proj_2d)
        plt.grid(True)
        plt.show()

    def clustering(self, low_dim_cells):
        sc_test_data = sc.read("data/sc_test_file.h5ad")
        sc_test_data_copy = sc_test_data.copy()
        sp_low_dim_cells = sp_sparse.csr_matrix(low_dim_cells)

        sc_ann_data = sc.AnnData(sp_low_dim_cells)

        sc_ann_data.obs_names = sc_test_data_copy.obs_names
        sc_ann_data.obsm = sc_test_data_copy.obsm

        self.plot_clustering(sc_test_data_copy, "Clustering with original test data")

        self.plot_clustering(sc_ann_data, "Clustering with low-dimensional test data")

        #print(sc_ann_data.obs_names)
        #sc_ann_data.obs_names = 
        #sc_ann_data.obs_names = np.repeat('fake', sc_ann_data.shape[0])
        #sc_ann_data.obs_names_make_unique()

        #sc.pl.umap(anndata_clustered, color=['leiden', 'CST3', 'NKG7'])
        # pre-processing
        #sc.pp.recipe_zheng17(clustered)
        #sc.tl.pca(clustered, n_comps=50)

        # clustering
        #sc.pp.neighbors(anndata_clustered, n_pcs=50)
        #sc.tl.louvain(anndata_clustered)
        #sc.pl.louvain(anndata_clustered)
        
    def plot_clustering(self, clustering_data, title="Clustering"):
    
        file_path = "figures/{}.pdf".format(title)
        print(file_path)
        
        ann_data = clustering_data.copy()

        sc.pp.neighbors(ann_data, n_neighbors=50, n_pcs=0, knn=False, use_rep='X', method='gauss')

        sc.tl.umap(ann_data)

        sc.tl.leiden(ann_data, key_added='clusters', resolution=1.0)
 
        sc.pl.umap(ann_data, color='clusters', add_outline=False, legend_loc='on data',
           legend_fontsize=8, legend_fontoutline=2, frameon=False, title=title, save=title)



if __name__ == "__main__":

    scVis = SCVisualise()

    data = scVis.get_features("data/output.csv")
    
    scVis.clustering(data)
    
    #scVis.plot_TSNE(data)
    
    #scVis.plot_UMAP(data)
