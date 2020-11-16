import os
import sys
import csv
import collections
import numpy as np
import scanpy as sc


class FilterSC(object):
    def __init__(self, **kwargs):
        self.dataset_dir = "data"
        self.mtx_file_path = os.path.join(self.dataset_dir, kwargs["mtx_file"])
        self.genes_file_path = os.path.join(self.dataset_dir, kwargs["genes_file"])
        self.obs_file_path = os.path.join(self.dataset_dir, kwargs["obs_file"])
        self.h5ad_file_path = os.path.join(self.dataset_dir, kwargs["h5ad_file"])
        self.genes = list()
        self.obs_names = list()
        self.sc_data = None

    def check_processed_file(self, processed_path):
        try:
            self.sc_data = sc.read(os.path.join(self.dataset_dir, processed_path))
            return True
        except Exception as ex:
            return False

    def read_files(self):
        with open(self.genes_file_path, "r") as gene_file:
            genes = csv.reader(gene_file, delimiter='\t')
            for gene in genes:
                self.genes.append(gene[1])

        with open(self.obs_file_path, "r") as obs_file:
            obs_names = csv.reader(obs_file, delimiter='\t')
            for ob in obs_names:
                self.obs_names.append(ob[0])
        # assign genes and obs
        print("Reading raw sc data...")
        self.sc_data = sc.read(self.mtx_file_path) 
        self.sc_data = self.sc_data.transpose()
        self.sc_data.var_names = self.genes
        self.sc_data.var_names_make_unique()
        self.sc_data.obs_names = self.obs_names
        self.sc_data.obs_names_make_unique()
        self.sc_data.write(filename=self.h5ad_file_path)

    def filter_single_cell(self):
        min_cells = 3
        min_genes = 10
        lib_size = 20000
        print("Filtering raw data...")
        sc.pp.filter_cells(self.sc_data, min_genes=min_genes, copy=False)
        sc.pp.filter_cells(self.sc_data, min_counts=min_cells, copy=False)
        print("Filtering of the raw data is done with minimum %d genes per cell and minimum %d cells." % (min_genes, min_cells))
        cells_count = self.sc_data.shape[0]
        genes_count = self.sc_data.shape[1]

        print("Cells number is %d , with %d genes per cell."
            % (cells_count, genes_count))

        print("Scaling raw data...")
        # scaling
        sc.pp.normalize_per_cell(self.sc_data, counts_per_cell_after=lib_size)

        print("Saving processed data...")
        output_h5ad_processed_file = "68kPBMCs_processed.h5ad"
        output_h5ad_processed_path = os.path.join(self.dataset_dir, output_h5ad_processed_file)
        self.sc_data.write(output_h5ad_processed_path)

    def create_train_data(self):
        split_share = 0.2
        samples = 100 # self.sc_data.shape[0]
        dimensions = self.sc_data.shape[1]
        n_split = int(split_share * samples)

        print("Creating training and test data...")

        train_data = np.zeros((samples - n_split, dimensions))
        test_data = np.zeros((n_split, dimensions))

        sc_train_data = self.sc_data[n_split:]
        sc_test_data = self.sc_data[:n_split]

        j = 0
        k = 0
        for i, row in enumerate(self.sc_data):
            df = row.to_df()
            df_list = df.values.tolist()
            if i >= samples:
                break
            if i < n_split:
                test_data[j] = df_list[0]
                j += 1
            else:
                train_data[k] = df_list[0]
                k += 1
        print("train data size: ({0}, {1})".format(train_data.shape[0], train_data.shape[1]))
        print("test data size: ({0}, {1})".format(test_data.shape[0], test_data.shape[1]))
        
        print("sc train data size: ({0}, {1})".format(sc_train_data.shape[0], sc_train_data.shape[1]))
        print("sc test data size: ({0}, {1})".format(sc_test_data.shape[0], sc_test_data.shape[1]))

        return train_data, test_data, sc_train_data, sc_test_data

        '''def __process_line(self, line):
        
        sct = collections.namedtuple('sc', ('barcode', 'count_no', 'genes_no'))
        scmd = sct(barcode=line.obs_names[0],
               count_no=int(np.sum(line.X)),
               genes_no=line.obs['n_genes'][0]
              )
        
        return line.X'''
