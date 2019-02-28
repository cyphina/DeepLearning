
#%%
#Import right modules and open the right files
import os
import pandas as pd
import matplotlib.pyplot as plt

def save_fig(figName, tightLayout = True):
    savePath = os.path.join(os.getcwd(), figName + ".png")
    print('Saving figure at ', savePath)
    if tightLayout:
        plt.tight_layout()
    plt.savefig(savePath, format='png', dpi=300)

#Load up the dataframes depending if we run it in the folder where this program is, or in the root project folder
if os.path.split(os.getcwd())[-1] == "Classification":
    saveResPath = os.path.join(os.getcwd().replace("Classification", ""), "Results", "NamLabelMixups")
else:
    saveResPath = os.path.join(os.getcwd(), "Results", "NamLabelMixups")

fileNames = os.listdir(saveResPath)
filesToAnalyze = []

for fileName in fileNames:
    if fileName.endswith("size400.csv"):
        filesToAnalyze.append(fileName)

filesToAnalyze

#%%
resultDescriptions = []
labelMixPer = []

#Get the mean values for each column in the table of trials
for file in filesToAnalyze:
    result = pd.read_csv(os.path.join(saveResPath, file))
    desc = result.describe()
    resultDescriptions.append(desc)
    labelMixPer.append(result.iloc[0,8])

result.columns, result.describe()

#%%
#Plot the accuracy
rowSize = 3
columnSize = 5
yNames = resultDescriptions[0].columns
totalGraphs = len(resultDescriptions[0].columns)
plt.figure(figsize=(25,15))

for r in range(totalGraphs):
    print(rowSize, columnSize, r+1)
    y = [result.iloc[1,r] for result in resultDescriptions]
    plt.subplot(rowSize, columnSize, r+1)
    plt.plot(labelMixPer, y, "g-")
    plt.title(str(yNames[r]) + " vs Percent Label Mixup (2000 instances)", fontsize=12)
    plt.ylabel(str(yNames[r]), fontsize=14, rotation=90)
    plt.xlabel("Perc of labels mixed up", fontsize=14)
    
save_fig('LabelMixup2000')   
plt.show()
