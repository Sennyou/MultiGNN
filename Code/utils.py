import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import random
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class load_data():
    def __init__(self, rna_file, atac_file, normalize=True):
        rna_df = pd.read_csv(rna_file, index_col=0)
        self.atac_df = pd.read_csv(atac_file, index_col=0)
        self.rna_df = rna_df.loc[self.atac_df.index]
        self.normalize = normalize

    def data_normalize(self, data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T

    def get_rna_feature(self):
        rna_feature = self.rna_df.values
        if self.normalize:
            rna_feature = self.data_normalize(rna_feature)

        rna_feature = rna_feature.astype(np.float32)

        return rna_feature

    def get_atac_feature(self):
        atac_feature = self.atac_df.values
        if self.normalize:
            atac_feature = self.data_normalize(atac_feature)
        atac_feature = atac_feature.astype(np.float32)
        return atac_feature

    def get_geneName(self):
        return self.rna_df.index

    def get_geneNum(self):
        return self.rna_df.shape[0]


def getnetwork(network_path, geneNames):
    network = pd.read_csv(network_path, sep=",")

    network = network[network["TF"].isin(geneNames.values) & network["target"].isin(geneNames.values)]
    # Remove self-loops.
    network = network[network.TF != network.target]
    # 去除重复
    network.drop_duplicates(keep='first', inplace=True)
    network["label"] = 1
    network = network.astype({'label': np.int32})
    network["TF"] = network["TF"].apply(lambda x: get_gene_index(geneNames, x))
    network["target"] = network["target"].apply(lambda x: get_gene_index(geneNames, x))
    return network


def get_neg_label(pos_lableSet, tfs, geneNames, neg_num):
    tfNum = len(tfs)
    geneNum = len(geneNames)
    neg_labelSet = pd.DataFrame({"TF": [], "target": [], "label": []})
    i = 0
    while i < neg_num:
        tf = random.randint(0, tfNum - 1)
        tf_num = get_gene_index(geneNames, tfs[tf])
        gene_num = random.randint(0, geneNum - 1)
        if geneNames[gene_num] in tfs or tf_num==gene_num:
            continue
        if pos_lableSet[(pos_lableSet["TF"] == tf_num) & (pos_lableSet["target"] == gene_num)].empty and (
                neg_labelSet[(neg_labelSet["TF"] == tf_num) & (neg_labelSet["target"] == gene_num)].empty
        ):
            neg_labelSet.loc[len(neg_labelSet)] = [tf_num, gene_num, 0]
            i = i + 1
    neg_labelSet = neg_labelSet.astype({'label': int})
    return neg_labelSet


def adj_generate(train_data, geneNum, direction=False, loop=False):
    adj = sp.csr_matrix((geneNum, geneNum), dtype=np.float32)
    for pos in train_data:
        tf = pos[0]
        target = pos[1]
        if direction:
            if pos[-1] == 1:
                adj[tf, target] = 1.0
        else:
            if pos[-1] == 1:
                adj[tf, target] = 1.0
                adj[target, tf] = 1.0
    if loop:
        adj = adj + sp.eye(geneNum)
    return adj


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def Evaluation(y_true, y_pred):
    prec, rec, thr = precision_recall_curve(y_true, y_pred)
    aupr_val = auc(rec, prec)
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    auc_val = auc(fpr, tpr)
    return auc_val, aupr_val, fpr, tpr, rec, prec



def split_Set(network, geneNames, tfs, networkDensity, test_rate=0.2):
    network = network.sample(frac=1).reset_index(drop=True)
    neg_num = int((1 - test_rate) * len(network) + test_rate * len(network) * (1 - networkDensity) / networkDensity)
    neg_labelSet = get_neg_label(network, tfs, geneNames, neg_num)

    train_neg = neg_labelSet[0:int((1 - test_rate) * len(network))]
    test_neg = neg_labelSet[int((1 - test_rate) * len(network)):]
    train_pos = network[0:int((1 - test_rate) * len(network))]
    test_pos = network[int((1 - test_rate) * len(network)):]
    

    train_data = pd.concat([train_pos, train_neg], ignore_index=True)
    test_data = pd.concat([test_pos, test_neg], ignore_index=True)
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    train_data = train_data.astype(np.int32)
    test_data = test_data.astype(np.int32)

    return test_data, train_data


def get_gene_index(geneNames, gene):
    for index, name in enumerate(geneNames):
        if gene == name:
            return index
    return -1

