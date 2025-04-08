import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
# from gears import PertData, GEARS
# from tears import PertData, GEARS
import wandb
import pandas as pd
import scanpy as sc
import plotly.express as px
import traceback
import sys

def download_pretrained(run):
	run.file(f'{run.id}/config.pkl').download('models', exist_ok=True)
	run.file(f'{run.id}/model.pt').download('models', exist_ok=True)

# Main function to initialize Hydra and parse the configuration
@hydra.main(config_name="config", config_path="config", version_base="1.3")
def main(cfg: DictConfig):

    try:
        train(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

def train(cfg: DictConfig):
    ablation_mode = cfg.ablation_mode
    dataset_name = cfg.dataset_name
    # dataset_name = "rep_gwp"
    model = cfg.model
    non_linearity = cfg.non_linearity
    cross_gene_module = cfg.cross_gene_module
    no_perturb = cfg.no_perturb
    epochs = cfg.epochs
    tags = cfg.tags
    gnn = cfg.gnn

    if model == "gears":
        from gears import PertData, GEARS
    else:
        raise ValueError("Model not found")

    pert_data = PertData(cfg.data_cache)

    pert_data.load(data_name=dataset_name)

    # specify data split
    pert_data.prepare_split(split = 'simulation', seed = 1)
    # get dataloader with batch size
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

    # set up and train a model
    gears_model = GEARS(pert_data, device = 'cuda', tags=tags, weight_bias_track=True, 
                     exp_name=f'{model}-ab{ablation_mode}-dn{dataset_name}-nl{non_linearity}-cm{cross_gene_module}-no_perturb{no_perturb}-gnn{gnn}' if not no_perturb else f"gears-no_perturb-dn{dataset_name}",proj_name='GSPP-publication',
                     seed=cfg.seed)

    gears_model.model_initialize(hidden_size = 64, ablation_mode = ablation_mode, non_linearity=non_linearity,
                              cross_gene_module= cross_gene_module, no_perturb = no_perturb, gnn=gnn)
    gears_model.train(epochs = epochs)

    # save/load model
    gears_model.save_model(f'gears-{ablation_mode}')
    # gears_model.load_pretrained('gears')

    if not no_perturb:

        api = wandb.Api()
        run = api.run(f"{gears_model.wandb.run.entity}/{gears_model.wandb.run.project}/{gears_model.wandb.run.id}")


        # set up and train a model
        # gears_model = GEARS(pert_data, device = 'cuda')
        download_pretrained(run)
        # gears_model.load_pretrained(f"models/{run.id}")
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
    main()
