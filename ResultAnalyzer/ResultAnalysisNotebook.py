
#%%
#Import right modules and open the right files
import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from enum import IntEnum

def save_fig(figName, tightLayout = True):
    savePath = os.path.join(os.getcwd(), figName + ".png")
    print('Saving figure at ', savePath)
    if tightLayout:
        plt.tight_layout()
    plt.savefig(savePath, format='png', dpi=300)

#Set the right cwd manually

class EMixStrategy(IntEnum):
  NoStrategy = 0
  Random = 1
  Most_Occuring=2
  Gradients=3
  Weights=4
  Furthest_Bounds = 5
  Furthest_Bounds_SVM = 6

cwdFolderName = os.path.split(os.getcwd())[-1]
#Load up the dataframes depending if we run it in the folder where this program is, or in the root project folder
if cwdFolderName == "Classification" or cwdFolderName == "ResultAnalyzer":
    saveResPath = os.path.join(os.getcwd().replace(cwdFolderName, ""), "Results", "NamLabelMixups")
else:
    saveResPath = os.path.join(os.getcwd(), "Results", "NamLabelMixups")

fileNames = os.listdir(saveResPath)

files = {}
for e in EMixStrategy:
    files[e] = []

for strategyFileName in [os.path.join(saveResPath, str(strategy)) for strategy in EMixStrategy]:
    if os.path.exists(strategyFileName):
        for fileName in os.listdir(strategyFileName):
            if fileName.endswith(str(EMixStrategy.Random) + ".csv"):
                files[EMixStrategy.Random].append(fileName)
            elif fileName.endswith(str(EMixStrategy.Most_Occuring) + ".csv"):
                files[EMixStrategy.Most_Occuring].append(fileName)
            elif fileName.endswith(str(EMixStrategy.Gradients) + ".csv"):
                files[EMixStrategy.Gradients].append(str(fileName))
            elif fileName.endswith(str(EMixStrategy.Weights) + ".csv"):
                files[EMixStrategy.Weights].append(str(fileName))  
            elif fileName.endswith(str(EMixStrategy.Furthest_Bounds) + ".csv"):
                files[EMixStrategy.Furthest_Bounds].append(str(fileName))
            elif fileName.endswith(str(EMixStrategy.Furthest_Bounds_SVM) + ".csv"):
                files[EMixStrategy.Furthest_Bounds_SVM].append(str(fileName))

print(files)

#%%
print(files.values())

#%%
#Each strategy index holds the results pertainng to that strategy
#The results are a list of dataframes
res = [[] for _ in range(len(EMixStrategy))]
#List of [0,0.1,0.2,...,1] but we set the values here in case one day somebody wants to change it
resLabels = [[] for _ in range(len(EMixStrategy))]

def getResults(mixStrat):
    for file in files[mixStrat]:
        result = pd.read_csv(os.path.join(saveResPath, str(mixStrat), file))
        desc = result.describe()
        res[mixStrat].append(desc)
        resLabels[mixStrat].append(result.iloc[0,6])

#Get the mean values for each column in the table of trials
for i, file in enumerate(files.values()):
    if len(file) > 0:
        getResults(EMixStrategy(i))

#%%
# i is strategy, j is percentage (0-100% with 10% increments)
res[1][0]

#%%
#Plot the accuracy
rowSize = 3
columnSize = 5
yNames = res[1][0].columns
totalGraphs = len(res[1][0].columns)
plt.figure(figsize=(25,15))

#Loop through all the columns (to make a graph for every statistic)
for r in range(totalGraphs):
    print(rowSize, columnSize, r+1)

    #Loop through the results for each strategy, extract mean for each column and plot it
    y = [None] * len(EMixStrategy)
    for i, row in enumerate(y):
        if len(res[i]) > 0:
            #Row 1 is the mean in case we have multiple values recorded for each mix percentage trial
            y[i] = [result.iloc[1,r] for result in res[i]] 
            plt.subplot(rowSize, columnSize, r+1)
            plt.plot(resLabels[i], y[i], "g-", label=EMixStrategy(i), color = 'C' + str(i))

    plt.title(str(yNames[r]) + " vs Percent Label Mixup (8000 instances)", fontsize=12)
    plt.ylabel(str(yNames[r]), fontsize=14, rotation=90)
    plt.xlabel("Perc of labels mixed up", fontsize=14)
    plt.legend(loc="upper right")
    
save_fig('LabelMixup8000')   
plt.show()


#%%
