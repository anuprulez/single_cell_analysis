import os
import csv
import scanpy.api as sc
import collections
import numpy as np



dataset_dir = "data/"
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
    #print(type(sc_genes))
    #print(dir(sc_genes))
    train_data[i] = sc_genes

print(train_data)   
print(train_data.shape)
