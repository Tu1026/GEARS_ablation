import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from tears import PertData, GEARS


# Main function to initialize Hydra and parse the configuration
def main():
    # Example function using the configuration
    train()

def train():
	ablation_mode = False
	dataset_name = 'norman'
	# dataset_name = 'replogle-gwp'
	non_linearity = True
	cross_gene_module= True
 
	# get data
	# pert_data = PertData('./data/curated_rxrx')
	pert_data = PertData('./data')
	# load dataset in paper: norman, adamson, dixit.
	pert_data.load(data_name = dataset_name)
	# specify data split
	pert_data.prepare_split(split = 'simulation', seed = 1)
	# get dataloader with batch size
	pert_data.get_dataloader(batch_size = 64, test_batch_size = 128)

	# set up and train a model
	gears_model = GEARS(pert_data, device = 'cuda', 
                     exp_name='tears-ab{ablation_mode}-dn{dataset_name}-nl{non_linearity}-cm{cross_gene_module}',
                     proj_name='TEARS')
	gears_model.model_initialize(hidden_size = 64, ablation_mode = ablation_mode, non_linearity=non_linearity,
                              cross_gene_module= cross_gene_module)
	gears_model.train(epochs = 20)

	# save/load model
	gears_model.save_model(f'gears-{ablation_mode}')
	# gears_model.load_pretrained('gears')

		
if __name__ == "__main__":
    main()
    
    
