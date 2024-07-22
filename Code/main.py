import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import argparse
from MultiGNN import MultiGNN
import scipy.sparse as sp
import tensorflow as tf
from utils import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of test set.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--epoch_num', type=int, default= 60, help='Number of epoch.')
parser.add_argument('--GCN_hidden_dim', type=int, default=512, help='Number of head attentions.')
parser.add_argument('--emb_dim', type=int, default=512, help='Alpha for the leaky_relu.')
parser.add_argument('--MLP_hidden_dim', type=int, default=128, help='The dimension of hidden layer')
parser.add_argument('--reduction',type=str,default='add', help='how to integrate feature')
parser.add_argument('--Lambda', type=float, default=0.5, help='The size of each batch')

args = parser.parse_args()
test_size = args.test_size
learning_rate = args.lr
epoch_num = args.epoch_num
GCN_hidden_dim = args.GCN_hidden_dim
emb_dim = args.emb_dim
MLP_hidden_dim = args.MLP_hidden_dim
reduction = args.reduction
Lambda = args.Lambda

rna_file = './Data/pbmc_rna.csv'
atac_file = './Data/pbmc_atac.csv'
tfs_file = './Data/TF_list.txt'
network_path = './Data/network_pbmc.csv'


print("...Start loading data...")
dataLoad = load_data(rna_file=rna_file, atac_file=atac_file)
rna_feature = dataLoad.get_rna_feature()
atac_feature = dataLoad.get_atac_feature()
print("rna shape:",rna_feature.shape)
print("atac shape:",atac_feature.shape)
geneName = dataLoad.get_geneName()
geneNum = dataLoad.get_geneNum()
print("Total number of gene:", geneNum)
print("...Reading TFs...")
tfs = pd.read_table(tfs_file, header=None)
print("Total number of tf:",len(tfs))
tfs = list(set(tfs.loc[:, 0].values) & set(geneName))
print("Number of TFs used for prediction after screening:",len(tfs))
print("...Reading network...")

network = getnetwork(network_path, geneName)
print("Number of gene pairs included in the network:",len(network))
networkDensity = (len(network) * 2) / (len(tfs) * len(rna_feature) - len(tfs))
print("network Densit:", networkDensity)
print("The test set size will be ",test_size)
print("...Start dividing the dataset...")
test_data, train_data = split_Set(network, geneName, tfs, networkDensity, test_size)
# train_data.to_csv("../Data/train_Breast_1000.csv", sep=",", index=False)
# test_data.to_csv("../Data/test_Breast_1000.csv", sep=",", index=False)


rna_feature = tf.constant(rna_feature, dtype=np.float32)
atac_feature = tf.constant(atac_feature, dtype=np.float32)
rna_norm = tf.nn.l2_normalize(rna_feature, 1)
atac_norm = tf.nn.l2_normalize(atac_feature, 1)
sample_num = rna_feature.shape[1]


adj = adj_generate(train_data=train_data, geneNum=geneNum, direction=True, loop=True)
rowsum = np.array(adj.sum(1))
degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
adj_normalized = degree_mat_inv_sqrt.dot(adj)
adj_normalized = adj_normalized.dot(degree_mat_inv_sqrt).tocoo()
biases = sparse_to_tuple(adj_normalized)


bias_in = tf.SparseTensor(indices=biases[0], values=biases[1], dense_shape=[geneNum, geneNum])
bias_in = tf.cast(bias_in, dtype=np.float32)
train_sample = tf.constant(train_data,dtype=np.int32)
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate, decay_steps=10, decay_rate=0.90)
optimizer = tf.keras.optimizers.Adam(exponential_decay)
mse = tf.keras.losses.MeanSquaredError()
model = MultiGNN(sample_num, GCN_hidden_dim, emb_dim, MLP_hidden_dim, reduction, Lambda)
print("...Start training...")

@tf.function(reduce_retracing=True)
def train_step(adj, train_sample, rna_norm, atac_norm):
    with tf.GradientTape() as tape:
        y_pred = model(adj, train_sample[:, 0:2], rna_norm, atac_norm)
        loss = mse(y_true=train_sample[:, 2], y_pred=y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

start = time.time()
for e in range(epoch_num):
    loss = train_step(bias_in, train_sample, rna_norm, atac_norm)
    print(f"Epoch {e+1}, Loss: {loss.numpy()}")

end = time.time()
print("Finish training, take time:", end-start)
print("...Start testing...")
test_sample = tf.constant(test_data, dtype=np.int32)
pred = model(bias_in, test_sample[:, 0:2], rna_norm, atac_norm)
rel = test_sample[:, 2]
auc_val, aupr_val, fpr, tpr, rec, prec = Evaluation(rel, pred)
print("modle performance:")
print("AUC:",auc_val)
print("AUPR:",aupr_val)
result = test_data.iloc[:,0:2]
result["pred"] = pred
result["TF"] = result["TF"].apply(lambda x: geneName[x])
result["target"] = result["target"].apply(lambda x: geneName[x])
result.to_csv("./output/result.csv", sep=",", index=False)
