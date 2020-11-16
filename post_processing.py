import os
import sys
import csv
import collections
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp_sparse


class SCPostProcessing(object):
    def __init__(self, sc_te_data, **kwargs):
        self.save_path = kwargs["output_file"]
        self.save_test_path = kwargs["sc_test_file"]
        self.sc_te_data = sc_te_data

    def save_results(self, pred_results):
        self.sc_te_data.write(self.save_test_path)
        dataframe = pd.DataFrame(pred_results)
        dataframe.to_csv(self.save_path, sep="\t", header=False, index=False, index_label=False)
        
    
        
        
        
