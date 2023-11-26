import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sys
import seaborn as sns
import time



cwd=os.getcwd()
path=cwd.replace('\\src','')

try:
    os.makedirs(path+'\\ConfusionMatrices\\')
except FileExistsError:
    # directory already exists
    pass

outputpath=path+'\\ConfusionMatrices\\'

path += '\\log_metrics\\'

#ADD FOLDER NAME HERE FOR NOW:
folder = 'targeted_T0T1N_total2_Max_mal1N_rounds1'
path += folder + '\\'
print(path)

#Sort through all files in the log_metrics folder
# select only those files related to a specific experiment
#for now, using the " attack" keyword, assuming multiple experiments of the same
#attack have not been run
maxround = int(folder.split('rounds')[1])-1
attack = folder.split('N_total')[0]

print(attack)

# Hard coded the classes here for cifar10 dataset
classes = list(range(0, 10))

selected_files=os.listdir(path)


stages={}
rounds=[]
for file in selected_files:

    round = file.split('Round')[1]
    round = round.split('_')[0]
    round = int(round)
    if round == maxround:
        pass
    else:
        continue

    data = pd.read_csv(path+file)
    fronthalf = file.split(attack)[0]
    fronthalf = fronthalf.replace('B','')
    fronthalf = fronthalf.replace('M', '')
    stage = int(fronthalf)


    if stage in stages:
        stages[stage][0] += list(data['y_true'].values)
        stages[stage][1] += list(data['y_pred'].values)
    else:
        stages[stage]=[list(data['y_true'].values), list(data['y_pred'].values)]

# for Each Stage Create Metrics

accuracy = []
precision = []
recall = []
f1 = []

target_accuracy = []
target_precision = []
target_recall = []
target_f1 = []

for key in stages.keys():
    print('num malicious clients', key)
    y_true = stages[key][0]
    y_pred = stages[key][1]

    #print(y_true)
    #print(y_pred)
    print(set(y_true))
    print(set(y_pred))

    cr = classification_report(y_true, y_pred, output_dict=True, zero_division=np.nan)
    print(cr)

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(df_cm, annot=True)

    ax.set_xlabel("Predicted Class", fontsize=14, labelpad=20)
    ax.set_ylabel("Actual Class", fontsize=14, labelpad=20)
    ax.set_title("Num_malicious"+str(key), fontsize=14, pad=20)

    try:
        os.makedirs(outputpath + folder)
    except FileExistsError:
        # directory already exists
        pass

    plt.savefig(outputpath + folder + '\\' + ' Num_malicious' + str(key) + '.png')
    #plt.show()

