import os
import pandas as pd

cwd=os.getcwd()
path=cwd.replace('\\src','')
path+='\\log_metrics\\'

print(path)

attack='random'

files=os.listdir(path)
selected_files=[]
for file in files:
    if attack in file:
        selected_files.append(file)
print(selected_files)

rounds={}
for file in selected_files:
    data=pd.read_csv(path+file)
    fronthalf=file.split('Round')[1]
    round=fronthalf.split('_')[0]
    round=int(round)
    if round in rounds:
        rounds[round][0].append(data['y_true'].values)
        rounds[round][1].append(data['y_pred'].values)
    else:
        rounds[round]=[list(data['y_true'].values), list(data['y_pred'].values)]