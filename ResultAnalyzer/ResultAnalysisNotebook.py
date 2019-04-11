
#%%
#Import right modules and open the right files
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

def save_fig(figName, tightLayout = True):
    savePath = os.path.join(os.getcwd(), figName + ".png")
    print('Saving figure at ', savePath)
    if tightLayout:
        plt.tight_layout()
    plt.savefig(savePath, format='png', dpi=300)

#Set the right cwd manually

class EMixStrategy(Enum):
  NoStrategy = 0
  Random = 1
  Most_Occuring=2
  Gradients=3
  Weights=4

cwdFolderName = os.path.split(os.getcwd())[-1]
#Load up the dataframes depending if we run it in the folder where this program is, or in the root project folder
if cwdFolderName == "Classification" or cwdFolderName == "ResultAnalyzer":
    saveResPath = os.path.join(os.getcwd().replace(cwdFolderName, ""), "Results", "NamLabelMixups")
else:
    saveResPath = os.path.join(os.getcwd(), "Results", "NamLabelMixups")

fileNames = os.listdir(saveResPath)

files = []
randomFiles = [] #Using random pick
occuringFiles = [] #Picking based on occurances of most popular features
gradientFiles = []

for strategyFileName in fileNames:
    for fileName in os.listdir(os.path.join(saveResPath, strategyFileName)): 
        if fileName.endswith(str(EMixStrategy.Random) + ".csv"):
            files[EMixStrategy.Random].append(fileName)
        elif fileName.endswith(str(EMixStrategy.Most_Occuring) + ".csv"):
            files[EMixStrategy.Most_Occuring].append(fileName)
        elif fileName.endswith(str(EMixStrategy.Gradients) + ".csv"):
            files[EMixStrategy.Gradients].append(str(fileName))        

randomFiles, occuringFiles, gradientFiles

#%%
res = []
resLabels = []

def getResults(files, ):
    for file in files:
        result = pd.read_csv(os.path.join(saveResPath, file))
        desc = result.describe()
        res.append(desc)
        mixPer.append(result.iloc[0,6])

#Get the mean values for each column in the table of trials
getResults(randomFiles, resultDescriptionsRand, labelMixPerRand)
getResults(randomFiles, resultDescriptionsRand, labelMixPerRand)
getResults(randomFiles, resultDescriptionsRand, labelMixPerRand)

#%%
#Plot the accuracy
rowSize = 3
columnSize = 5
yNames = resultDescriptionsRand[0].columns
totalGraphs = len(resultDescriptionsRand[0].columns)
plt.figure(figsize=(25,15))

for r in range(totalGraphs):
    print(rowSize, columnSize, r+1)
    y1 = [result.iloc[1,r] for result in resultDescriptionsRand]
    y2 = [result.iloc[1,r] for result in resultDescriptionsOccuring]
    plt.subplot(rowSize, columnSize, r+1)
    plt.plot(labelMixPerRand, y1, "g-", label="Random")
    plt.plot(labelMixPerOccur, y2, "r-", label="MostOccuring")
    plt.title(str(yNames[r]) + " vs Percent Label Mixup (8000 instances)", fontsize=12)
    plt.ylabel(str(yNames[r]), fontsize=14, rotation=90)
    plt.xlabel("Perc of labels mixed up", fontsize=14)
    plt.legend(loc="upper right")
    
save_fig('LabelMixup8000')   
plt.show()


#%%
