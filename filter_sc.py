import os
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
        samples = 200 # self.sc_data.shape[0]
        dimensions = self.sc_data.shape[1]
        n_split = int(split_share * samples)

        print("Creating training data...")

        train_data = np.zeros((n_split, dimensions))
        test_data = np.zeros((samples - n_split, dimensions))

        for i, line in enumerate(self.sc_data):
            sc_genes = __process_line(line)
            if i > n_split:
                train_data[i] = sc_genes
            else:
                 test_data[i] = sc_genes
            if i > samples:
                break
        print(train_data.shape)
        print(test_data.shape)

    def __process_line(line):
        '''
        sct = collections.namedtuple('sc', ('barcode', 'count_no', 'genes_no'))
        scmd = sct(barcode=line.obs_names[0],
               count_no=int(np.sum(line.X)),
               genes_no=line.obs['n_genes'][0]
              )
        '''
        return line.X






'''dataset_dir = "data/"
data_file = "matrix.mtx"
var_names_file = "genes.tsv"
obs_names_file = "barcodes.tsv"
output_h5ad_file = "68kPBMCs.h5ad"

data_path = os.path.join(dataset_dir,data_file)
var_names_path = os.path.join(dataset_dir,var_names_file)
obs_names_path = os.path.join(dataset_dir,obs_names_file)
output_h5ad_path = os.path.join(dataset_dir,output_h5ad_file)


with open(var_names_path, "r") as var_file:
    var_read = csv.reader(var_file, delimiter='\t')
    var_names = []
    for row in var_read:
        var_names.append(row[1])
        

with open(obs_names_path, "r") as obs_file:
    obs_read = csv.reader(obs_file, delimiter='\t')
    obs_names = []
    for row in obs_read:
        obs_names.append(row[0])
        
print("Reading raw data...")
        
andata = sc.read(data_path) 
andata = andata.transpose()

andata.var_names = var_names
andata.var_names_make_unique()
andata.obs_names = obs_names
andata.obs_names_make_unique()

print("Writing processed data...")

andata.write(filename=output_h5ad_path)


# filtering 

print("Filtering raw data...")

min_cells = 3,
min_genes = 10

sc_raw = andata

sc.pp.filter_cells(sc_raw, min_genes=min_genes, copy=False)
print("Filtering of the raw data is done with minimum "
      "%d genes per cell." % min_genes)

sc.pp.filter_genes(sc_raw, min_cells=min_cells, copy=False)
print("Filtering of the raw data is done with minimum"
      " %d cells per gene." % min_cells)

cells_count = sc_raw.shape[0]
genes_count = sc_raw.shape[1]

print("Cells number is %d , with %d genes per cell."
      % (cells_count, genes_count))
      
print("Scaling raw data...")

# scaling
scale = "normalize_per_cell_LS_20000"

if "normalize_per_cell_LS_" in str(scale):

    lib_size = int(scale.split('_')[-1])
    sc.pp.normalize_per_cell(sc_raw,
                             counts_per_cell_after=lib_size)
    scale = {"scaling": 'normalize_per_cell_LS',
                  "scale_value": lib_size}

else:

    warnings.warn("The scaling of the data is unknown, library size "
                  "library size normalization with 20k will be applied")

    lib_size = int(sscale.split('_')[-1])
    sc.pp.normalize_per_cell(sc_raw,
                             counts_per_cell_after=lib_size)
    self.scale = {"scaling": 'normalize_per_cell_LS',
                  "scale_value": lib_size}

print("Scaling of the data is done using " + scale["scaling"]
      + " with " + str(scale["scale_value"]))
      

print("Saving processed data...")

output_h5ad_processed_file = "68kPBMCs_processed.h5ad"
output_h5ad_processed_path = os.path.join(dataset_dir, output_h5ad_processed_file)
sc_raw.write(output_h5ad_processed_path)


sct = collections.namedtuple('sc', ('barcode', 'count_no', 'genes_no'))
exp_share = 0.2
n_exp_data = int(exp_share * sc_raw.shape[0])
sc_raw_exp = sc_raw[:n_exp_data]

print("Creating training data...")

train_data = np.zeros((sc_raw_exp.shape[0], sc_raw_exp.shape[1]))

def process_line(line):
    scmd = sct(barcode=line.obs_names[0],
               count_no=int(np.sum(line.X)),
               genes_no=line.obs['n_genes'][0]
              )
    return line.X, scmd


for i, line in enumerate(sc_raw_exp):
    sc_genes, d = process_line(line)
    train_data[i] = sc_genes

print(train_data)   
print(train_data.shape)'''
