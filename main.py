import os
import sys
import time

import numpy as np
import autoencoder
import filter_sc


class SCA(object):
    """ 
    """
    def filter_sc(self, **kwargs):
        sc = filter_sc.FilterSC(**kwargs)
        sc_data = sc.check_processed_file(kwargs["processed_file"])
        if not sc_data:
            sc.read_files()
            sc.filter_single_cell()
        return sc.create_train_data()
       
    def get_data(self):
        generated_data = np.random.randn(self.dim[0], self.dim[1])
        test_data = np.random.randn(100, self.dim[1])
        return generated_data, test_data
        
    def train_ae(self, input_data, test_data, sc_train_data, sc_test_data):
        i_shape = input_data.shape[1]
        s_time = time.time()
        ae = autoencoder.SCAutoEncoder(input_dim=i_shape).train_model(input_data, test_data, sc_train_data, sc_test_data)
        e_time = time.time()
        print("Training and prediction finished in {} seconds".format(int(e_time - s_time)))

if __name__ == "__main__":

    #init_dim = (1000, 96)
    
    sca = SCA()

    # filter single-cell data
    
    tr_data, te_data, sc_tr_data, sc_te_data = sca.filter_sc(mtx_file="matrix.mtx", genes_file="genes.tsv", obs_file="barcodes.tsv", h5ad_file="68kPBMCs.h5ad", processed_file= "68kPBMCs_processed.h5ad")
    
    #tr_data, te_data = sca.get_data()
    sca.train_ae(tr_data, te_data, sc_tr_data, sc_te_data)
