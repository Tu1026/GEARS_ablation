import scanpy as sc
import numpy as np
import torch
from torch_geometric.data import Data

def parallel_create_cell_graph_dataset(data_path, pert_category, pert_names, ctrl_adata_path,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs

        Parameters
        ----------
        data_path: str
            Data path
        pert_category: str
            Perturbation category
        num_samples: int
            Number of samples to create per perturbed cell (i.e. number of
            control cells to map to each perturbed cell)

        Returns
        -------
        list
            List of cell graphs

        """

        num_de_genes = 20     
        # adata_ = split_adata 
        split_adata = sc.read_h5ad(data_path)
        ctrl_adata = sc.read_h5ad(ctrl_adata_path)
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        if 'rank_genes_groups_cov_all' in adata_.uns:
            de_genes = adata_.uns['rank_genes_groups_cov_all']
            de = True
        else:
            de = False
            num_de_genes = 1
        Xs = []
        ys = []

        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            # Get the indices of applied perturbation
            pert_idx = get_pert_idx(pert_category, pert_names)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition_name'][0]
            if de:
                de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            else:
                de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                # Use samples from control as basal expression
                ctrl_samples = ctrl_adata[np.random.randint(0,
                                        len(ctrl_adata), num_samples), :]
                for c in ctrl_samples.X:
                    Xs.append(c)
                    ys.append(cell_z)

        # When considering a control perturbation
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                Xs.append(cell_z)
                ys.append(cell_z)

        # Create cell graphs
        cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(create_cell_graph(X.toarray(),
                                y.toarray(), de_idx, pert_category, pert_idx))

        return cell_graphs


def get_pert_idx(pert_category, pert_names):
	"""
	Get perturbation index for a given perturbation category

	Parameters
	----------
	pert_category: str
		Perturbation category

	Returns
	-------
	list
		List of perturbation indices

	"""
	try:
		pert_idx = [np.where(p == pert_names)[0][0]
				for p in pert_category.split('+')
				if p != 'ctrl']
	except:
		print(pert_category)
		pert_idx = None
		
	return pert_idx

def create_cell_graph(X, y, de_idx, pert, pert_idx=None):
	"""
	Create a cell graph from a given cell

	Parameters
	----------
	X: np.ndarray
		Gene expression matrix
	y: np.ndarray
		Label vector
	de_idx: np.ndarray
		DE gene indices
	pert: str
		Perturbation category
	pert_idx: list
		List of perturbation indices

	Returns
	-------
	torch_geometric.data.Data
		Cell graph to be used in dataloader

	"""

	feature_mat = torch.Tensor(X).T
	if pert_idx is None:
		pert_idx = [-1]
	return Data(x=feature_mat, pert_idx=pert_idx,
				y=torch.Tensor(y), de_idx=de_idx, pert=pert)