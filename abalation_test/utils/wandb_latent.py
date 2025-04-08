import wandb
from tears import PertData, GEARS
import scanpy as sc
import plotly.express as px
import pandas as pd
import submitit


def main(run_id):
	def download_pretrained(run):
		run.file(f'{run.id}/config.pkl').download('models', exist_ok=True)
		run.file(f'{run.id}/model.pt').download('models', exist_ok=True)

	wandb.init(project="GEARS", id= run_id, resume='must')
	api = wandb.Api()
	run = api.run(f"ding-group/GEARS/{run_id}")
	if 'pert_variational' in run.tags:
		from tears import PertData, GEARS
	else:
		from gears import PertData, GEARS
          
     
	# get data
	# pert_data = PertData('./data')
	# load dataset in paper: norman, adamson, dixit.
	# pert_data.load(data_name = run.config['dataset'])
	dataset_name = run.config['dataset']
	if dataset_name == 'rep_gwp':
		pert_data = PertData('./data/curated_rxrx')
		pert_data.load(data_path = '/mnt/ps/home/CORP/wilson.tu/project/transcriptomic/GEARS/abalation_test/data/curated_rxrx/rep_gwp')
	else:
		pert_data = PertData('./data')
		pert_data.load(data_name = dataset_name)
	# specify data split
	pert_data.prepare_split(split = 'simulation', seed = 1)
	# get dataloader with batch size
	pert_data.get_dataloader(batch_size = 64, test_batch_size = 128)

	# set up and train a model
	gears_model = GEARS(pert_data, device = 'cuda')
	download_pretrained(run)
	gears_model.load_pretrained(f"models/{run.id}")
	gears_model.model.eval()
	gears_model.model(next(iter(gears_model.dataloader['train_loader'])).to('cuda'))


	## Perturbation embeddings
	lst = []
	list(map(lambda x: lst.append(x.pert_idx) , list(gears_model.dataloader["train_loader"])))
	print()
	pert_embeddings = gears_model.model.pert_global_emb.reshape(gears_model.num_perts, -1)
	pert_ad = sc.AnnData(pert_embeddings.to('cpu').numpy(), obs= gears_model.pert_list)
	sc.pp.neighbors(pert_ad, use_rep='X', n_neighbors=30, metric='cosine')
	sc.tl.umap(pert_ad)
	plot_df_pert = pd.DataFrame(pert_ad.obsm['X_umap'])
	plot_df_pert['label'] = pert_ad.obs.iloc[:, 0].values
	# plot_df_pert['color']= pd.Series(pert_data.pert_names).isin(pd.Series(pert_data.adata.obs.condition.unique()).astype('str').str.split('+').explode())
	plot_df_pert['color'] = None
	plot_df_pert.loc[pd.Series(lst).explode().explode().unique()[pd.Series(lst).explode().explode().unique() >0], "color"] = 'perturbed'
	plot_df_pert.loc[~plot_df_pert.index.isin(pd.Series(lst).explode().explode().unique()[pd.Series(lst).explode().explode().unique() >0]), "color"] = 'not perturbed'
	plot_df_pert = plot_df_pert.rename(columns={0: 'x', 1: 'y'})
	fig1 = px.scatter(plot_df_pert, x='x', y='y', hover_data=['label'],color='color')
	fig1.update_traces(marker=dict(size=2,))
	table_perturb= wandb.Table(data=plot_df_pert)
	wandb.log({"Perturbation latent plot": fig1})
	wandb.log({"Perturbation latent table": table_perturb})

	## Gene embeddings
	gene_embeddings = gears_model.model.base_emb.reshape(64, gears_model.num_genes, -1)
	ad = sc.AnnData(gene_embeddings[0].to('cpu').numpy(), obs= gears_model.adata.var)
	sc.pp.neighbors(ad, use_rep='X', n_neighbors=30, metric='cosine')
	sc.tl.umap(ad)
	plot_df_gene = pd.DataFrame(ad.obsm['X_umap'])
	plot_df_gene['label'] = ad.obs.gene_name.values
	plot_df_gene = plot_df_gene.rename(columns={0: 'x', 1: 'y'})
	fig2 = px.scatter(plot_df_gene, x='x', y='y', hover_data=['label'],)
	fig2.update_traces(marker=dict(size=2,))
	table_gene= wandb.Table(data=plot_df_gene)

	wandb.log({"Gene latent plot": fig2})
	wandb.log({"Gene latent table": table_gene})

 
	# ## Final prediction embedding
	# outs = []
	# pert_
	# for batch in gears_model.dataloader["train_loader"]:
	# 	gears_model.model(batch.to("cuda"))
		
	# out_embeddings = gears_model.model.out.reshape(64, gears_model.num_genes, -1)
	# ad = sc.AnnData(out_embeddings[0].to("cpu").numpy(), obs=gears_model.adata.var)
	# sc.pp.neighbors(ad, use_rep="X", n_neighbors=30, metric="cosine")
	# sc.tl.umap(ad)
	# plot_df_gene = pd.DataFrame(ad.obsm["X_umap"])
	# plot_df_gene["label"] = ad.obs.gene_name.values
	# plot_df_gene = plot_df_gene.rename(columns={0: "x", 1: "y"})
	# fig2 = px.scatter(
	# 	plot_df_gene,
	# 	x="x",
	# 	y="y",
	# 	hover_data=["label"],
	# )
	# fig2.update_traces(
	# 	marker=dict(
	# 		size=2,
	# 	)
	# )
	# table_gene = wandb.Table(data=plot_df_gene)

	# wandb.log({"Final embedding plot": fig2})
	# wandb.log({"Final embedding table": table_gene})

	wandb.finish()


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="log_folder")
    executor.update_parameters(
	timeout_min=300,   # job timeout in minutes
	partition="hopper",  # change to your SLURM partition
	gpus_per_node=1,  # set if using GPUs
	slurm_mem="200G",   # adjust based on memory needs
)

    api = wandb.Api()
    executor.map_array(main, [run.id for run in api.runs("ding-group/GEARS") if run.id != 'tshr3zwr'])
    # executor.map_array(main, [run.id for run in api.runs("ding-group/GEARS") if run.id != ])
    # executor.map_array(main, [run.id for run in api.runs("ding-group/GEARS") if run.id == 'tshr3zwr'])

