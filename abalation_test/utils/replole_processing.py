import scanpy as sc
import pandas as pd
from gears import PertData
from scipy import sparse
import submitit

def main():
	adata = sc.read_h5ad("/mnt/ps/home/CORP/wilson.tu/project/transcriptomic/GEARS/abalation_test/data/curated_rxrx/replogle_2022.h5ad")
	sc.pp.normalize_total(adata)
	sc.pp.log1p(adata)
	sc.pp.highly_variable_genes(adata,n_top_genes=8000, subset=True)
	adata.obs['gene_name'] = adata.obs['gene_name'].astype(str)

	adata.obs.loc[adata.obs['is_control'], 'gene_name'] = 'ctrl'
	adata.obs.loc[~adata.obs['is_control'], 'gene_name'] = adata.obs.loc[~adata.obs['is_control'], 'gene_name'] + "+ctrl"
	adata.obs['gene_name'] = adata.obs['gene_name'].astype("category")
	adata.obs['condition'] = adata.obs['gene_name']


	# adata = sc.read_h5ad("/mnt/ps/home/CORP/wilson.tu/project/transcriptomic/GEARS/abalation_test/data/curated_rxrx/perturb_processed.h5ad")
	adata.X = sparse.csr_matrix(adata.X)
	adata.write_h5ad("/mnt/ps/home/CORP/wilson.tu/project/transcriptomic/GEARS/abalation_test/data/curated_rxrx/perturb_processed.h5ad")

	# get data
	pert_data = PertData('./data/curated_rxrx')
	pert_data.new_data_process(dataset_name = 'rep_gwp', adata = adata) # specific dataset name and adata object
	pert_data.load(data_path = './data/curated_rxrx/rep_gwp') # load the processed data, the path is saved folder + dataset_name
	pert_data.prepare_split(split = 'simulation', seed = 1) # get data split with seed
	# specify data split
	# get dataloader with batch size
	pert_data.get_dataloader(batch_size = 64, test_batch_size = 128)


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="log_folder")
    executor.update_parameters(
	timeout_min=90,   # job timeout in minutes
	partition="hopper",  # change to your SLURM partition
	gpus_per_node=1,  # set if using GPUs
	slurm_mem="300G",   # adjust based on memory needs
)

    executor.map_array(main, [1])