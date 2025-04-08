import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from gears import PertData, GEARS


# Main function to initialize Hydra and parse the configuration
@hydra.main(config_name="config", config_path="config", version_base="1.3")
def main(cfg: DictConfig):

    # Example function using the configuration
    train(cfg)

def train(cfg: DictConfig):
	ablation_mode = cfg.ablation_mode
	dataset_name = cfg.dataset_name
	non_linearity = cfg.non_linearity
	cross_gene_module= cfg.cross_gene_module
 
	# get data
	pert_data = PertData('./data')
	pert_data.load(data_name = dataset_name)
	# pert_data = PertData('./data/curated_rxrx')
	# load dataset in paper: norman, adamson, dixit.
	# pert_data.load(data_path = '/mnt/ps/home/CORP/wilson.tu/project/transcriptomic/GEARS/abalation_test/data/curated_rxrx/rep_gwp')
	# specify data split
	pert_data.prepare_split(split = 'simulation', seed = 1)
	# get dataloader with batch size
	pert_data.get_dataloader(batch_size = 64, test_batch_size = 128)

	# set up and train a model
	gears_model = GEARS(pert_data, device = 'cuda', weight_bias_track=True, 
                     exp_name=f'gears-ab{ablation_mode}-dn{dataset_name}-nl{non_linearity}-cm{cross_gene_module}')
	gears_model.model_initialize(hidden_size = 64, ablation_mode = ablation_mode, non_linearity=non_linearity,
                              cross_gene_module= cross_gene_module)
	gears_model.train(epochs = 20)

	# save/load model
	gears_model.save_model(f'gears-{ablation_mode}')
	# gears_model.load_pretrained('gears')

		
if __name__ == "__main__":
    main()