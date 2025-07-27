# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:37:34 2019

@author: Semiu
"""

import networkx as nx
#from node2vec import Node2Vec
from scipy.io import loadmat
#import scipy
"""
Reading the Homo_sapiens.mat into a data variable
"""
data = loadmat('Homo_sapiens.mat')
net = data['network']
grp = data['group']

data2arrray = data.toarray()

#data2graph = loadmat('Homo_sapiens.mat')
"""
Converting the Homo_sapiens.mat file to a Graph
"""
#graph = nx.from_scipy_sparse_matrix(data2graph, create_using=nx.MultiGraph())
graph = nx.Graph(data2arrray)


import scipy.io
import pandas as pd

mat = scipy.io.loadmat('Homo_sapiens.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
data.to_csv("homosapiens.csv")

"""
Using plot
Another method to analyse the CAP Curve involves reading the plot we generated above. Steps for the same are:
1.	Draw a vertical line at 50% from the x-axis till it crosses the Support Vector Classifier plot.
2.	At the point, where the vertical line cuts the trained model, draw a horizontal line such that it cuts the y-axis.
3.	Calculate the percentage of class 1 identified with respect to the total count of class 1 labels.
Once we know the percentage, we can use the following brackets to analyse it:
1. Less than 60%: Rubbish Model
2. 60% — 70%: Poor Model
3. 70% — 80%: Good Model
4. 80% — 90%: Very Good Model
5. More than 90%: Too Good to be True
Note that if the value is more than 90%, it’s a good practice to test for over fitting.
First, I find the index by calculating int value of 50% of total test data. I use it to plot a vertical dashed line (— — —) from this point to the trained model. Next, I plot the line from this point of intersection to the y-axis. I determine the percentage by dividing the class 1.0 values observed till now with the total class 1.0 data points and multiplying it by 100. I get the value as 93.55%.

"""
import matplotlib.pyplot as plt
#import numpy as np 
import seaborn as sns
import pandas as pd


EpV = pd.DataFrame(columns=['Reward', 'Epsilon Values', 'No of Slot'])

# Append rows in Empty Dataframe by adding dictionaries
EpV = EpV.append({'Reward': 8053, 'Epsilon Values': 0.02, 'No of Slot': '5 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 6986, 'Epsilon Values': 0.02, 'No of Slot': '10 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 8031, 'Epsilon Values': 0.04, 'No of Slot': '5 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 7393, 'Epsilon Values': 0.04, 'No of Slot': '10 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 8108, 'Epsilon Values': 0.06, 'No of Slot': '5 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 7525, 'Epsilon Values': 0.06, 'No of Slot': '10 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 7921, 'Epsilon Values': 0.08, 'No of Slot': '5 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 7371, 'Epsilon Values': 0.08, 'No of Slot': '10 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 7745, 'Epsilon Values': 0.10, 'No of Slot': '5 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 7261, 'Epsilon Values': 0.10, 'No of Slot': '10 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 3499, 'Epsilon Values': 0.12, 'No of Slot': '5 slots'}, ignore_index=True)
EpV = EpV.append({'Reward': 4148, 'Epsilon Values': 0.12, 'No of Slot': '10 slots'}, ignore_index=True)


plt.title('Epsilon-Greedy (Number of iterations = 1000)')
ax = sns.scatterplot(x='Epsilon Values', y='Reward', hue= 'No of Slot', data=EpV)

"""
Epsilon-Greedy (Number of iterations = 1000)
"""
x1 = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
y1 = [0.00, 8053.00, 8031.00, 8108.00, 7921.00, 7745.00, 3499.00]
y2 = [0.00, 6986.00, 7393.00, 7525.00, 7371.00, 7261.00, 4148.00]

fig, ax = plt.subplots()
ax.plot(x1, y1, 'r', label='5 slots')
ax.plot(x1, y2, 'g', label ='10 slots')

legend = ax.legend(loc='lower right', shadow=False, fontsize='small')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C0')
plt.xlabel('Epsilon Values')
plt.ylabel ('Reward')
plt.title('Epsilon-Greedy (Number of iterations = 1000)')
plt.show()
"""
Epsilon-Greedy (Epsilon Value: 0.1)
"""
x2 = [0, 200, 400, 600, 800, 1000, 1200]
y3 = [0, 1604, 3175, 4735, 6174, 7745, 9360]
y4 = [0, 1032, 2603, 4152, 5701, 7261, 8821]

fig, ax = plt.subplots()
ax.plot(x2, y3, 'r', label='5 slots')
ax.plot(x2, y4, 'b', label ='10 slots')

legend = ax.legend(loc='lower right', shadow=False, fontsize='small')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C0')
plt.xlabel('Number of Iterations')
plt.ylabel ('Reward')
plt.title('Epsilon-Greedy (Epsilon value: 0.1)')
plt.show()
"""
Annealing Epsilon-Greedy
"""
y5 = [0, 1494, 2999, 4493, 5943, 7503, 9096]
y6 = [0, 1472, 2955, 4405, 5877, 7426, 8975]

fig, ax = plt.subplots()
ax.plot(x2, y5, 'r', label='5 slots')
ax.plot(x2, y6, 'b', label ='10 slots')

legend = ax.legend(loc='lower right', shadow=False, fontsize='small')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C0')
plt.xlabel('Number of Iterations')
plt.ylabel ('Reward')
plt.title('Annealing Epsilon-Greedy')
plt.show()

"""
PSO Epsilon-Greedy
"""
y7 = [0, 1560, 1349, 2062, 5767, 7349, 8766]
y8 = [0, 856, 1613, 2601, 3336, 4148, 5851]
fig, ax = plt.subplots()
ax.plot(x2, y7, 'r', label='5 slots')
ax.plot(x2, y8, 'b', label ='10 slots')

legend = ax.legend(loc='lower right', shadow=False, fontsize='small')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C0')
plt.xlabel('Number of Iterations')
plt.ylabel ('Reward')
plt.title('PSO Epsilon-Greedy')
plt.show()
"""
Annealing Epsilon Greedy and Epsilon-Greedy (Number of Slot machine = 10)
"""
x3 = [0, 200, 400, 600, 800, 1000, 1200, 1400]

y9 = [0, 1153, 2724, 4251, 5844, 7525, 9118, 10634]
y10 = [0, 1472, 2955, 4405, 5877, 7426, 8975, 10370]
fig, ax = plt.subplots()
ax.plot(x3, y9, 'r', label='Epsilon Greedy')
ax.plot(x3, y10, 'b', label ='Annealing Epsilon Greedy')

legend = ax.legend(loc='lower right', shadow=False, fontsize='small')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C0')
plt.xlabel('Number of Iterations')
plt.ylabel ('Reward')
plt.title('Annealing Epsilon Greedy and Epsilon-Greedy (Number of Slot machine = 10)')
plt.show()
"""

"""
# Creating an empty Dataframe with column names as 
EpG = pd.DataFrame(columns=['Reward', 'Iteration', 'No of Slot'])

# Append rows in Empty Dataframe by adding dictionaries
EpG = EpG.append({'Reward': 1560, 'Iteration': 200, 'No of Slot': '5 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 856, 'Iteration': 200, 'No of Slot': '10 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 1349, 'Iteration': 400, 'No of Slot': '5 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 1613, 'Iteration': 400, 'No of Slot': '10 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 2062, 'Iteration': 600, 'No of Slot': '5 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 2601, 'Iteration': 600, 'No of Slot': '10 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 5767, 'Iteration': 800, 'No of Slot': '5 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 3336, 'Iteration': 800, 'No of Slot': '10 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 7349, 'Iteration': 1000, 'No of Slot': '5 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 4148, 'Iteration': 1000, 'No of Slot': '10 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 8766, 'Iteration': 1200, 'No of Slot': '5 slots'}, ignore_index=True)
EpG = EpG.append({'Reward': 5851, 'Iteration': 1200, 'No of Slot': '10 slots'}, ignore_index=True)


plt.title('PSO Epsilon-Greedy')
ax = sns.scatterplot(x='Iteration', y='Reward', hue= 'No of Slot', data=EpG)





plt.plot('Iteration','Reward')
plt.xlabel('Iteration')
plt.ylabel ('Reward')




ax = sns.lineplot(x='Iteration', y='Reward', hue= 'No of Slot', data=EpG)
ax = sns.lineplot(x='Iteration', y='Reward', hue= 'No of Slot', data=EpG)
sns.barplot(x='Iteration', y='Reward', data=EpG)