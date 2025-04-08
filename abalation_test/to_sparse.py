import scanpy as sc
from scipy.sparse import csr_matrix
import submitit
import wandb
def main():
	adata = sc.read_h5ad("/mnt/ps/home/CORP/wilson.tu/project/transcriptomic/GEARS/abalation_test/data/curated_rxrx/rep_gwp/perturb_processed.h5ad")
	sparse = csr_matrix(adata.X)
	adata.X = sparse
	adata.write_h5ad("/mnt/ps/home/CORP/wilson.tu/project/transcriptomic/GEARS/abalation_test/data/curated_rxrx/rep_gwp/perturb_processed_sparse.h5ad")


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="log_folder")
    executor.update_parameters(
	timeout_min=300,   # job timeout in minutes
	slurm_partition="def",  # change to your SLURM partition
	gpus_per_node=1,  # set if using GPUs
	slurm_mem="800G",   # adjust based on memory needs
)

    executor.submit(main)


	