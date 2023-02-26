# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:52:21 2019

@author: Semiu: Network Analysis Program for Adv. Data Mining Assignment 3
"""
"""
Importing neccesary libaries
"""
import pandas as pd
import powerlaw
import networkx as nx 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import csv
"""
Reading the Homo_sapiens.mat (with variables network and group) file:
a. loadmat is imported from scipy.io
b. the loadmat function is used to load the Homo_sapiens.mat and assigned to x variable
c. the two variables 'network' and 'group' are assigned to net and grp respectively.
"""
x = loadmat('Homo_sapiens.mat')
net = x['network']
grp = x['group']
"""
Reading the variable network from the Homo_sapiens.mat file, and 
assigned to new variable named "net".
"""
net = loadmat('Homo_sapiens.mat')['network']
"""
Converting the network variable, net, to a Graph
"""
N = nx.Graph(net)
"""
Checking whether the Network, N, is connected or not. It returns "False". 
This implies it is not connected.
"""
print(nx.is_connected(N))
"""
Finding out the number of the connected components of the network N by:
a. Getting the list of the connected components and printing the information
b. Getting the size of the list
"""
num = [len(c) for c in sorted(nx.connected_components(N), key=len, reverse=True)]
print(num)
len(num)
"""
Degeree of Centrality of the graph calculated as dc
"""
dc = nx.degree_centrality(N)
"""
Closeness Centrality of the graph calculated as cc
"""
cc = nx.closeness_centrality(N)
"""
Betweenness Centrality of the graph calculated as bc
"""
bc = nx.betweenness_centrality(N)
"""
Harmonic Centrality of the graph calculated as hc
"""
hc = nx.harmonic_centrality(N)
"""
Eigenvector Centrality of the graph calculated as ec
"""
ec = nx.eigenvector_centrality(N)
"""
Clustering Coefficient  of the graph calculated as coeff
"""
coeff = nx.clustering(N)
"""
Page Rank  of the graph calculated as pr
"""
pr = nx.pagerank(N)
"""
The values of each of the measures calculated are written into a csv file named 
measures.csv. These are computed into single values for each of the respective measures
"""
with open ("measures.csv", "w+") as measures:
    for i in range (len(pr)):
        measure = ""
        measure += str(dc[i])
        measure +=  "," + str(cc[i])
        measure += "," + str(bc[i])
        measure += "," + str(hc[i])
        measure += "," + str(ec[i])
        measure += "," + str(coeff[i])
        measure += ","+ str(pr[i])
        measure += ("\n")
        measures.write(measure)
"""
The measures.csv file has a new line to serve as its headings. These are the labels for 
the respective measures calculated.
The csv file is subsequently read into a df variable
"""
with open('measures.csv',newline='') as file:
    read = csv.reader(file)
    data = [line for line in read]
with open('measures.csv','w',newline='') as file:
    w = csv.writer(file)
    w.writerow(['dc', 'cc', 'bc','hc','ec','coeff','pr'])
    w.writerows(data)
df = pd.read_csv("measures.csv", sep=",") 
"""
To show the row-by-row arrangment of the data, with the measures as columnd headings.
The first five rows of the measures.csv file is shown.
"""
df.head(5)
"""
The histograms for each of the measures are displayed:
a. Histogram for Degree Centrality
"""
df['dc'].plot.hist(grid=True, bins=15, rwidth=0.9,color='#70db70')
plt.title('Histogram for Degree Centrality')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
"""
b. Histogram for Closeness Centrality
"""
df['cc'].plot.hist(grid=True, bins=15, rwidth=0.9,color='#1a1aff')
plt.title('Histogram for Closeness Centrality')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
"""
c. Histogram for Betweenness Centrality
"""
df['bc'].plot.hist(grid=True, bins=15, rwidth=0.9,color='#ff3385')
plt.title('Histogram for Betweenness Centrality')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
"""
d. Histogram for Harmonic Centrality
"""
df['hc'].plot.hist(grid=True, bins=15, rwidth=0.9,color='#ffff66')
plt.title('Histogram for Harmonic Centrality')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
"""
e. Histogram for Eigenvector Centrality
"""
df['ec'].plot.hist(grid=True, bins=15, rwidth=0.9,color='#cc0000')
plt.title('Histogram for Eigenvector Centrality')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
"""
f. Histogram for Clustering Coefficient
"""
df['coeff'].plot.hist(grid=True, bins=15, rwidth=0.9,color='#ff6600')
plt.title('Histogram for Clustering Coefficient')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
"""
g. Histogram for PageRank
"""
df['pr'].plot.hist(grid=True, bins=15, rwidth=0.9,color='#e600e6')
plt.title('Histogram for PageRank')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
"""
Fitting the Degree of Centrality into Power Law
"""
dc_values = df['dc'].values
results = powerlaw.Fit(dc_values)
"""
Visualizing the Power Law-fitted Degree Centality using PDF
"""
plt.title('Power Law-fitted Degree Centality using PDF')
results.plot_pdf(color='b', linewidth=2)
"""
Visualizing the Power Law-fitted Degree Centality using CCDF
"""
plt.title('Power Law-fitted Degree Centality using CCDF')
results.plot_ccdf(color='g', linewidth=2)

"""
Visualizing the goodness of the power law fit using the PDF 
"""
plt.title('Goodness of the power law fit using the PDF')
fig4 = results.plot_pdf(color='b', linewidth=2)
results.power_law.plot_pdf(color='b', linestyle='--', ax=fig4)
"""
Visualizing the goodness of the power law fit using the CCDF 
"""
plt.title('Goodness of the power law fit using the CCDF')
fig5 = results.plot_ccdf(color='r', linewidth=2)
results.power_law.plot_ccdf(color='r', linestyle='--', ax=fig5)