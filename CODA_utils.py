# graph-CODA utils

# distance table  :  category distance : same class=0; different classes=1
# graph augmentation : number of edges

import sys
import os
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp

import dgl
import torch
import torch.nn.functional as F
from sklearn.model_selection import ShuffleSplit
from copy import deepcopy


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def cate_grouping(labels):
    group = {}
    num_classes = labels.max().int() + 1
    for i in range(num_classes):
        group[i] = torch.nonzero(labels == i, as_tuple=True)[0]

    return group


def class_matrix(g, outputs, test_mask, labels):

    pro_outputs = F.log_softmax(outputs, dim=1)
    pro_labels = pro_outputs.max(dim=1)[1].type_as(labels)
    l = deepcopy(labels)
    labels[test_mask] = pro_labels[test_mask]

    n = len(g.nodes())
    group = cate_grouping(labels)
    cat_dis = torch.ones(n, n).to(labels.device)
    for i in range(n):
        cate = labels[i].item()
        cat_dis[i] = torch.index_fill(cat_dis[i], dim=0, index=group[cate], value=0)
    return cat_dis, l


def graph_augmentation(m, k):
    n = m.shape[0]
    # m = torch.triu(m, diagonal=1)
    m = m.flatten()

    top_index = torch.topk(m, k, largest=False)
    top_index0 = top_index.indices // n
    top_index1 = top_index.indices % n
    g = dgl.graph((top_index0, top_index1), num_nodes=n)
    g = dgl.add_reverse_edges(g)
    return g


def sparse_graph_augmentation(g, labels, k):
    group = cate_grouping(labels)
    G = dgl.DGLGraph()
    G = dgl.add_nodes(G, len(g.nodes()))
    # src, dst = g.edges()
    n = len(g.nodes())

    index_0 = []
    index_1 = []
    for i in range(n):
        cate = labels[i].item()
        index_i = torch.ones(len(group[cate]), dtype=torch.int64)
        index_i = torch.index_fill(index_i, dim=0, index=index_i, value=i)
        index_0.extend(index_i)
        index_1.extend(group[cate])
        G = dgl.add_edges(G, index_0, index_1)
        if len(G.edges()[0]) >= k:
            break
    return G


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset_str='cora'):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])
    D1_ = np.array(adj.sum(axis=1)) ** (-0.5)
    D2_ = np.array(adj.sum(axis=0)) ** (-0.5)
    D1_ = sp.diags(D1_[:, 0], format='csr')
    D2_ = sp.diags(D2_[0, :], format='csr')
    A_ = adj.dot(D1_)
    A_ = D2_.dot(A_)

    D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    A = sparse_mx_to_torch_sparse_tensor(A)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return A, features, labels, idx_train, idx_val, idx_test, adj


file_dir = "./data/new_data/"
def load_data_webkb(dataset_name, splits_file_path, train_percentage, val_percentage):
    graph_adjacency_list_file_path = os.path.join(f'{file_dir}', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join(file_dir, dataset_name,
                                                            f'out1_node_feature_label.txt')
    labels = []
    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        graph_node_features_and_labels_file.readline()
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            labels.append(int(line[2]))
            feat_dim = len(np.array(line[1].split(','), dtype=np.uint8))

    features = torch.zeros(len(labels), feat_dim)
    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        graph_node_features_and_labels_file.readline()
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            features[int(line[0])] = torch.from_numpy(np.array(line[1].split(','), dtype=np.uint8))

    index_0 = []
    index_1 = []
    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            index_0.append(int(line[0]))
            index_1.append(int(line[1]))
    g = dgl.graph((index_0, index_1), num_nodes=len(labels))

    adj = g.adj(scipy_fmt='csr')
    features = normalize_features(features)
    labels = np.array(labels)

    if splits_file_path:
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
    else:
        train_and_val_index, test_index = next(
            ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                np.empty_like(labels), labels))
        train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
            np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
        train_index = train_and_val_index[train_index]
        val_index = train_and_val_index[val_index]

        train_mask = np.zeros_like(labels)
        train_mask[train_index] = 1
        val_mask = np.zeros_like(labels)
        val_mask[val_index] = 1
        test_mask = np.zeros_like(labels)
        test_mask[test_index] = 1

    return g, adj, features, labels, train_mask, val_mask, test_mask


