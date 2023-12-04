import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sys
import seaborn as sns
import time



cwd=os.getcwd()
path=cwd.replace('\\src','')

try:
    os.makedirs(path+'\\AdditionalMetrics\\')
except FileExistsError:
    # directory already exists
    pass

outputpath=path+'\\AdditionalMetrics\\'

path += '\\log_metrics\\'

#ADD FOLDER NAME HERE FOR NOW:
folder = 'targeted_T1T2N_total4_Max_mal4N_rounds50'
path += folder + '\\'
print(path)

#Sort through all files in the log_metrics folder
# select only those files related to a specific experiment
#for now, using the " attack" keyword, assuming multiple experiments of the same
#attack have not been run
maxround = int(folder.split('rounds')[1])-1
attack = folder.split('N_total')[0]

print(attack, 'max round', maxround)

# Hard coded the classes here for cifar10 dataset
classes = list(range(0, 10))

selected_files=os.listdir(path)


stages={}
rounds=[]
for file in selected_files:

    round = file.split('Round')[1]
    round = round.split('_')[0]
    round = int(round)
    if round != maxround:
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

overall_accuracy = []
target_class_precision = []
target_class_recall = []
flipped_label_precision = []
flipped_label_recall = []

targeted = False
if 'targeted' in folder:
    back_half = folder.split('_T')[1]
    splits = back_half.split('T')
    target_class = splits[0]
    target_class = int(target_class)
    flipped_label = splits[1].split('N_')[0]
    flipped_label = int(flipped_label)
    targeted = True

for key in stages.keys():
    print('num malicious clients', key)
    y_true = stages[key][0]
    y_pred = stages[key][1]

    #print(y_true)
    #print(y_pred)
    print(set(y_true))
    print(set(y_pred))

    cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    overall_accuracy.append(accuracy_score(y_true, y_pred))
    #print(cr)
    if targeted:
        target_class_precision.append( cr[str(target_class)]['precision'])
        target_class_recall.append( cr[str(target_class)]['recall'])
        #Remove the original label and switch the labels
        y_targ = [flipped_label if x == target_class else -1 for x in y_true]
        # [4 if x==1 else x for x in a]
        cr_targ = classification_report(y_targ, y_pred, output_dict=True, zero_division=0)
        #print(cr_targ)
        flipped_label_precision.append( cr_targ[str(flipped_label)]['precision'])
        flipped_label_recall.append(    cr_targ[str(flipped_label)]['recall'])

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

    plt.savefig(outputpath + folder + '\\' + 'CM_Num_malicious' + str(key) + '.png')
    plt.close()
    #plt.show()

if targeted:
    print(overall_accuracy)
    target_class_precision[:] = [0 if math.isnan(x) else x for x in target_class_precision]
    print(target_class_precision)
    print(target_class_recall)


    print(flipped_label_precision)
    print(flipped_label_recall)
    x = stages.keys()
    print(x)
    plt.clf()
    plt.plot(x, overall_accuracy, label="overall_accuracy", marker='o', linestyle='dashed')
    plt.plot(x, target_class_precision, label="target_class_precision")
    plt.plot(x, target_class_recall, label="target_class_recall")
    plt.plot(x, flipped_label_precision, label="flipped_label_precision")
    plt.plot(x, flipped_label_recall, label="flipped_label_recall")
    plt.legend()

    plt.savefig(outputpath + folder + '\\' + 'TargetedMetrics'+'.png')
    plt.show()

