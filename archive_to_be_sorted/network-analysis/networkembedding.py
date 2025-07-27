# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:48:10 2019

@author: Semiu: Network Embedding Program for Adv. Data Mining -Assignment 4
"""
import networkx as nx
from node2vec import Node2Vec
from scipy.io import loadmat
import matplotlib.pyplot as plt
#import scipy
"""
Reading the Homo_sapiens.mat into a data variable
"""
data = loadmat('Homo_sapiens.mat')
net = data['network']
grp = data['group']

data2graph = loadmat('Homo_sapiens.mat')['network']
"""
Converting the Homo_sapiens.mat file to a Graph
"""
#graph = nx.from_scipy_sparse_matrix(data2graph, create_using=nx.MultiGraph())
graph = nx.Graph(data2graph)
"""
Precompute probabilities and generate walks
"""
graph_node2vec = Node2Vec(graph, dimensions=2, walk_length=30, num_walks=200, workers=1)
"""
Generate Nodes, in line with the node2vec constructor
"""
graph_model = graph_node2vec.fit(window=10, min_count=1, batch_words=4)

"""
Scatter Plot Visualizatio of the graph_node2vec
"""
colors = (0, 1)
plt.scatter(graph_node2vec, c=colors, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()