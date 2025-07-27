# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:34:02 2019

@author: Semiu
"""

"""
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


"""
#Code performance Testing
#Start
#1
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#1
x1 = [100.00, 1000.00, 10000.00, 100000.00, 1000000.00]
y1 = [1.00, 3.00, 3.00, 17.00, 46.00]
y2 = [1.00, 1.00, 4.00, 8.00, 23.00]

fig, ax = plt.subplots()
ax.plot(x1, y1, 'r', label='computeSumA')
ax.plot(x1, y2, 'g', label ='computeSumB')

legend = ax.legend(loc='lower right', shadow=False, fontsize='small')

plt.xlabel('Number of Iterations')
plt.ylabel ('Time Taken')
plt.title('computeSum - Based on Iterations')
plt.show()

#2
x2 = [20.00, 25.00, 30.00, 35.00, 40.00]
y3 = [0.00, 2.00, 2.00, 1.00, 1.00]
y4 = [1.00, 1.00, 1.00, 1.00, 21.00]

fig, ax = plt.subplots()
ax.plot(x2, y3, 'r', label='FindTestValueA')
ax.plot(x2, y4, 'g', label ='FindTestValueB')

legend = ax.legend(loc='upper left', shadow=False, fontsize='small')

plt.xlabel('Num of Cycles')
plt.ylabel ('Time Taken')
plt.title('computeSum - Based on Iterations')
plt.show()

#3 Bar chart
plt.style.use('ggplot')

horiz = ['FindTestValueA', 'FindTestValueB']
avgTime = [1.2, 1.0]

horiz_pos = [i for i, _ in enumerate(horiz)]

plt.bar(horiz_pos, avgTime, color='blue')
plt.xlabel("find()")
plt.ylabel("Average Time")
plt.title("Bar chart for the Average Time for FindtestValue")

plt.xticks(horiz_pos, horiz)

plt.show()






#end

"""


#Epsilon-Greedy (Epsilon Value: 0.1)
"""
#import matplotlib.pyplot as plt
#import numpy as np 
#import seaborn as sns
#import pandas as pd
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

"""
#Annealing Epsilon-Greedy

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

"""

#PSO Epsilon-Greedy

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

"""
#Annealing Epsilon Greedy and Epsilon-Greedy (Number of Slot machine = 10)

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