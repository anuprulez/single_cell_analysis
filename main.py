
import os
import sys

import numpy as np
import autoencoder


class SCA(object):
    """ 
    """
    def __init__(self, dim):
        self.dim = dim
       
    def get_data(self):
        generated_data = np.random.randn(self.dim[0], self.dim[1])
        test_data = np.random.randn(100, self.dim[1])
        return generated_data, test_data
        
    def train_ae(self, input_data, test_data):
        i_shape = input_data.shape[1]
        ae = autoencoder.SCAutoEncoder(input_shape=i_shape).setup_training(input_data, test_data)

if __name__ == "__main__":

    init_dim = (1000, 96)
    sca = SCA(init_dim)
    tr_data, te_data = sca.get_data()
    sca.train_ae(tr_data, te_data)
