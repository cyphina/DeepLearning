
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
    saveResPath = os.path.join(os.getcwd().replace("Classification", ""), "Results", "NamLabelMixups", "3212019")
else:
    saveResPath = os.path.join(os.getcwd(), "Results", "NamLabelMixups","3212019")

fileNames = os.listdir(saveResPath)

randomFiles = [] #Using random pick
occuringFiles = [] #Picking based on occurances of most popular features

for fileName in fileNames:
    if fileName.endswith("Random.csv"):
        randomFiles.append(fileName)
    elif fileName.endswith("Most_Occuring.csv"):
        occuringFiles.append(fileName)

randomFiles, occuringFiles

#%%
resultDescriptionsRand, resultDescriptionsOccuring = [],[]
labelMixPerRand, labelMixPerOccur = [],[]

#Get the mean values for each column in the table of trials
for file in randomFiles:
    result = pd.read_csv(os.path.join(saveResPath, file))
    desc = result.describe()
    resultDescriptionsRand.append(desc)
    labelMixPerRand.append(result.iloc[0,6])

for file in occuringFiles:
    result = pd.read_csv(os.path.join(saveResPath, file))
    desc = result.describe()
    resultDescriptionsOccuring.append(desc)
    labelMixPerOccur.append(result.iloc[0,6])

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
