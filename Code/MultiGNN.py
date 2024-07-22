import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
import numpy as np
def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim))
    initial = tf.random.uniform([input_dim, output_dim], minval = -init_range, maxval = init_range, dtype = tf.float32)
    return tf.Variable(initial, name=name)

class MultiGNN(tf.keras.Model):
    def __init__(self, sample_num, GCN_hidden_dim1=800, emb_dim=200, MLP_hidden_dim=64, reduction='add', Lambda=0.5,**kwargs):
        super().__init__(**kwargs)

        self.rna_GCN1 = GraphConv(sample_num, GCN_hidden_dim1, name="GCN_rna1")
        self.rna_GCN2 = GraphConv(GCN_hidden_dim1, emb_dim, name="GCN_rna2")

        self.reduction = reduction
        self.Lambda = Lambda

        self.atac_GCN1 = GraphConv(sample_num, GCN_hidden_dim1, name="GCN_atac1")
        self.atac_GCN2 = GraphConv(GCN_hidden_dim1, emb_dim, name="GCN_atac2")

        if self.reduction == 'attention':
            self.attention = Attention(emb_dim)
            self.predictor = MLPPredictor(emb_dim * 2, hidden_dim=MLP_hidden_dim)
        elif self.reduction == 'add':
            self.predictor = MLPPredictor(emb_dim * 2, hidden_dim=MLP_hidden_dim)
        elif self.reduction == 'concate':
            self.predictor = MLPPredictor2(emb_dim * 4, hidden_dim=MLP_hidden_dim)

    @tf.function(reduce_retracing=True)
    def call(self, adj, train_sample, rna, atac=None):
        h1 = self.rna_GCN1(adj, rna)
        h1 = self.rna_GCN2(adj, h1)
        h2 = self.atac_GCN1(adj, atac)
        h2 = self.atac_GCN2(adj, h2)
        if self.reduction == 'attention':
            h = tf.stack([h1, h2], axis=1)
            h, w = self.attention(h)
            pred = self.predictor(h, train_sample)
        elif self.reduction == 'add':
            h = tf.math.add(tf.multiply(h1, (1.0-self.Lambda)), tf.multiply(h2, self.Lambda))
            pred = self.predictor(h, train_sample)
        elif self.reduction == 'concate':
            pred = self.predictor(h1, h2, train_sample)
        pred = tf.nn.relu(pred)
        return pred


class Attention(tf.keras.layers.Layer):
    def __init__(self, emb_dim, hidden_dim=64, act=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        self.act = act
        self.w1 = weight_variable_glorot(emb_dim, hidden_dim, name='attention_w1')
        self.w2 = weight_variable_glorot(hidden_dim, 1, name='attention_w2')

    @tf.function(reduce_retracing=True)
    def call(self, input):
        x = input  
        x = tf.matmul(x, self.w1)
        x = self.act(x)
        x = tf.matmul(x, self.w2)  
        w = tf.nn.softmax(x, axis=1)
        return tf.math.reduce_sum(w * input, 1), w


class GraphConv(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, name, act=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        self.act = act
        self.w = weight_variable_glorot(input_dim, output_dim, name=name + '_gcn_w')

    @tf.function(reduce_retracing=True)
    def call(self, adj, input):
        x = input
        x = tf.matmul(x, self.w)
        output = tf.sparse.sparse_dense_matmul(adj, x)
        return self.act(output)


class MLPPredictor(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=64, act=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        self.act = act
        self.w1 = weight_variable_glorot(input_dim, hidden_dim, name='mlp_w1')
        self.b1 = tf.Variable(
            tf.zeros([hidden_dim]), name='mlp_b1')
        self.w2 = weight_variable_glorot(hidden_dim, 1, name='mlp_w2')

    @tf.function(reduce_retracing=True)
    def call(self, input, train_sample):
        tf_emb = tf.gather(input, train_sample[:, 0])
        target_emb = tf.gather(input, train_sample[:, 1])
        h = tf.concat([tf_emb, target_emb], 1)
        x = h
        x = tf.matmul(x, self.w1) + self.b1
        x = self.act(x)
        output = tf.matmul(x, self.w2) 
        return output



class MLPPredictor2(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=64, act=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        self.act = act
        self.w1 = weight_variable_glorot(input_dim, hidden_dim, name='mlp_w1')
        self.b1 = tf.Variable(
            tf.zeros([hidden_dim]), name='mlp_b1')
        self.w2 = weight_variable_glorot(hidden_dim, 1, name='mlp_w2')

    @tf.function(reduce_retracing=True)
    def call(self, input1, input2, train_sample):
        tf_emb1 = tf.gather(input1, train_sample[:, 0])
        target_emb1 = tf.gather(input1, train_sample[:, 1])
        tf_emb2 = tf.gather(input2, train_sample[:, 0])
        target_emb2 = tf.gather(input2, train_sample[:, 1])
        h = tf.concat([tf_emb1, target_emb1, tf_emb2, target_emb2], 1)
        x = h
        x = tf.matmul(x, self.w1) + self.b1
        x = self.act(x)
        output = tf.matmul(x, self.w2)
        return output
    
