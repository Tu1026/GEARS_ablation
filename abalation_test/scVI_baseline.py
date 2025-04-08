import scanpy as sc
from gears import PertData
import pandas as pd
import scgen
import wandb
from lightning.pytorch.loggers import WandbLogger
import submitit
from lightning.pytorch.callbacks import ModelCheckpoint

def main():
	dataset= 'replogle_k562_essential'
	# Initialize a new W&B run	

	wandb_logger = WandbLogger(project="scgen_baseline", name=f"scgen{dataset}")
	checkpoint_callback = ModelCheckpoint()
	# get data
	pert_data = PertData('./data')
	# load dataset in paper: norman, adamson, dixit.
	pert_data.load(dataset)
	# specify data split
	pert_data.prepare_split(split = 'simulation', seed = 1)
	# get dataloader with batch size
	pert_data.get_dataloader(batch_size = 64, test_batch_size = 128)



	train_conditions = pd.Series(list(set(pert_data.set2conditions['train'])))
	test_conditions = pd.Series(list(set(pert_data.set2conditions['test'])))
	val_conditions = pd.Series(list(set(pert_data.set2conditions['val'])))

	train_adata = pert_data.adata[pert_data.adata.obs.condition.isin(train_conditions)].copy()
	test_adata = pert_data.adata[pert_data.adata.obs.condition.isin(test_conditions)].copy()
	val_adata = pert_data.adata[pert_data.adata.obs.condition.isin(val_conditions)].copy()

	train_adata = train_adata.concatenate(val_adata)
	scgen.SCGEN.setup_anndata(train_adata, batch_key="condition", labels_key='cell_type')

	model = scgen.SCGEN(train_adata)
	model.save(f"models/scgen/{dataset}/model_perturbation_prediction.pt", overwrite=True)

	model.train(
		max_epochs=200,
		batch_size=32,
  		early_stopping_monitor='reconstruction_loss_validation',
		early_stopping=True,
		early_stopping_patience=25,
		logger=wandb_logger,
		callbacks=[checkpoint_callback],
  enable_checkpointing= True
	)
 
	model.save(f"models/scgen/{dataset}/model_perturbation_prediction.pt", overwrite=True)
 

	wandb.save(f"models/scgen/{dataset}")

	wandb.finish()
 
if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="log_folder")
    executor.update_parameters(
	timeout_min=90,   # job timeout in minutes
	partition="hopper",  # change to your SLURM partition
	gpus_per_node=1,  # set if using GPUs
	slurm_mem="300G",   # adjust based on memory needs
)

    executor.submit(main)