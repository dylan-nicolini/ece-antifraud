from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
import dgl
import random
import os
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

DATADIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "data/")


def featmap_gen(tmp_df=None):
    time_span = [2, 3, 5, 15, 20, 50, 100, 150,
                 200, 300, 864, 2590, 5100, 10000, 24000]
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []

    for trans_idx, trans_feat in tqdm(tmp_df.iterrows()):
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount

        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]

            new_df[f'trans_at_avg_{tname}'] = correct_data['Amount'].mean()
            new_df[f'trans_at_totl_{tname}'] = correct_data['Amount'].sum()
            new_df[f'trans_at_std_{tname}'] = correct_data['Amount'].std()
            new_df[f'trans_at_bias_{tname}'] = temp_amt - correct_data['Amount'].mean()
            new_df[f'trans_at_num_{tname}'] = len(correct_data)
            new_df[f'trans_target_num_{tname}'] = len(correct_data.Target.unique())
            new_df[f'trans_location_num_{tname}'] = len(correct_data.Location.unique())
            new_df[f'trans_type_num_{tname}'] = len(correct_data.Type.unique())

        post_fe.append(new_df)

    return pd.DataFrame(post_fe)


def sparse_to_adjlist(sp_matrix, filename):
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()

    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)

    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_risk_neighs(graph, risk_label=1):
    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata['label'][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)
    return torch.Tensor(ret)


def feat_map(graph, edge_feat):
    tensor_list = []

    for idx in tqdm(range(graph.num_nodes())):
        neighs = graph.predecessors(idx)
        tensor = torch.FloatTensor([
            edge_feat[neighs, 0].sum().item(),
            edge_feat[neighs, 1].sum().item(),
        ])
        tensor_list.append(tensor)

    return torch.stack(tensor_list)


if __name__ == "__main__":

    set_seed(42)

    # ------------------------------------------------------
    #  S-FFSD ONLY
    # ------------------------------------------------------

    print("processing S-FFSD data...")

    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSD.csv'))
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)

    data.to_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'), index=None)
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'))

    data = data.reset_index(drop=True)

    alls = []
    allt = []
    pair = ["Source", "Target", "Location", "Type"]

    for column in pair:
        src, tgt = [], []
        edge_per_trans = 3

        for _, c_df in tqdm(data.groupby(column), desc=column):
            c_df = c_df.sort_values(by="Time")
            df_len = len(c_df)
            sorted_idxs = c_df.index

            src.extend([sorted_idxs[i] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
            tgt.extend([sorted_idxs[i + j] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])

        alls.extend(src)
        allt.extend(tgt)

    g = dgl.graph((np.array(alls), np.array(allt)))

    for col in pair:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]

    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).long()
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).float()

    dgl.data.utils.save_graphs(DATADIR + "graph-S-FFSD.bin", [g])

    # ------------------------------------------------------
    #  Neighbor Risk Features (S-FFSD ONLY)
    # ------------------------------------------------------

    print("Generating neighbor risk-aware features for S-FFSD...")

    graph = dgl.load_graphs(DATADIR + "graph-S-FFSD.bin")[0][0]

    degree_feat = graph.in_degrees().unsqueeze(1).float()
    risk_feat = count_risk_neighs(graph).unsqueeze(1).float()
    edge_feat = torch.cat([degree_feat, risk_feat], dim=1)

    features_neigh = feat_map(graph, edge_feat)
    features_neigh = torch.cat((edge_feat, features_neigh), dim=1).numpy()
    features_neigh[np.isnan(features_neigh)] = 0.

    features_neigh = pd.DataFrame(features_neigh)
    scaler = StandardScaler()
    features_neigh = pd.DataFrame(
        scaler.fit_transform(features_neigh))

    features_neigh.to_csv(
        DATADIR + "S-FFSD_neigh_feat.csv", index=False)

    print("S-FFSD processing complete.")
