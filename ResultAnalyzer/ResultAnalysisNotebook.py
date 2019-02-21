
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
meanAccuracy = []
labelMixPer = []

#Get the mean values for each column in the table of trials
for file in filesToAnalyze:
    result = pd.read_csv(os.path.join(saveResPath, file))
    desc = result.describe()
    meanAccuracy.append(desc.iloc[1,0])
    labelMixPer.append(result.iloc[0,8])

#%%
#Plot the accuracy
plt.plot(labelMixPer, meanAccuracy, "g-")
plt.title("Accuracy vs Percent Label Mixup (400 instances)")
plt.ylabel("Accuracy", fontsize=14, rotation=90)
plt.xlabel("Perc of labels mixed up", fontsize=14)
plt.axis([-0.02,1.02, 0, 1])
save_fig('LabelMixupAccuracy400')
plt.show()
