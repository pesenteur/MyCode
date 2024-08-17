import numpy as np
import torch

def load_graph_data():
    adj_matrix = np.load('Data/adj_matrix.npy')
    adj_matrix_sparse = sp.coo_matrix(adj_matrix)
    edge_index, _ = from_scipy_sparse_matrix(adj_matrix_sparse)
    return edge_index
