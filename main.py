import os
import sys

import numpy as np
import autoencoder
import filter_sc


class SCA(object):
    """ 
    """
    def __init__(self, dim):
        self.dim = dim
        
    def filter_sc(self, **kwargs):
        sc = filter_sc.FilterSC(**kwargs)
        sc.read_files()
        sc.filter_single_cell()
        tr_data, te_data = sc.create_train_data()
        return tr_data, te_data
       
    def get_data(self):
        generated_data = np.random.randn(self.dim[0], self.dim[1])
        test_data = np.random.randn(100, self.dim[1])
        return generated_data, test_data
        
    def train_ae(self, input_data, test_data):
        i_shape = input_data.shape[1]
        ae = autoencoder.SCAutoEncoder(input_dim=i_shape).train_model(input_data, test_data)

if __name__ == "__main__":

    init_dim = (1000, 96)
    sca = SCA(init_dim)
    
    # filter single-cell data
    tr_data, te_data = sca.filter_sc(mtx_file="matrix.mtx", genes_file="genes.tsv", obs_file="barcodes.tsv", h5ad_file="68kPBMCs.h5ad")
    
    #tr_data, te_data = sca.get_data()
    sca.train_ae(tr_data, te_data)
