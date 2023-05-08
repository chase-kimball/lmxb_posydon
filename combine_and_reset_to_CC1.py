import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse 
#FIXME Automate Cuts
parser = argparse.ArgumentParser()
parser.add_argument('--infolder', type=str, required=True)
parser.add_argument('--outfile', type=str, required=True)
parser.add_argument('--Nbatches', type=int, default=100)

args = parser.parse_args()

for ii in range(1,args.Nbatches):
    print(ii)
    file_name = "{}/batches/batch_{}/population_{}.h5".format(args.infolder,str(ii),str(ii))
    oneline = pd.read_hdf(file_name,key='oneline')
    history = pd.read_hdf(file_name,key='history')
    print(len(oneline))
    
    i_CC1 = np.where((history.event=='CC1') & (history.S2_mass < 3) & (history.S1_mass>3))[0]

    index_CC1 = history.iloc[i_CC1].index
    
    history_CC1 = history.loc[index_CC1]
    oneline_CC1 = oneline.loc[index_CC1]

    #Make 15_50 Cut
    oneline_CC1 = oneline_CC1.loc[(oneline_CC1.S1_mass_i<50) & (oneline_CC1.S1_mass_i>15)]
    ind_cut = oneline_CC1.index
    history_CC1 = history_CC1.loc[ind_cut]

    for key in oneline_CC1.columns:
        if 'natal' in key:
            oneline_CC1.drop(columns=[key], inplace=True)
    copy_history = history_CC1.copy()
    curr_index = -1
    postSN = True
    history_CC1.reset_index(inplace=True)
    oneline_CC1.reset_index(inplace=True)

    i_CC1 = []
    for jj, row in history_CC1.iterrows():
        if not row.binary_index == curr_index:
            postSN = False
            i=0
            curr_index = row.binary_index

        if postSN: 
            history_CC1.drop([jj],inplace=True)

        if row.event == 'CC1':
 
            postSN = True


    old_indices = oneline_CC1.binary_index
    new_indices = np.arange(0,len(old_indices))
    map_index = dict(zip(old_indices,new_indices))
    oneline_CC1.binary_index = new_indices
    oneline_CC1.set_index('binary_index',inplace=True)  

    history_CC1.replace({'binary_index':map_index},inplace=True)
    history_CC1.set_index('binary_index',inplace=True)  
    
    oneline_CC1.index = oneline_CC1.index + int((ii-1)*100_000)
    history_CC1.index = history_CC1.index + int((ii-1)*100_000)
    
    if ii == 1:
        oneline_CC1_all = oneline_CC1.copy()
        history_CC1_all = history_CC1.copy()
    
    else:
        oneline_CC1_all = pd.concat([oneline_CC1_all, oneline_CC1])
        history_CC1_all = pd.concat([history_CC1_all, history_CC1])
        
history_CC1_all.reset_index(inplace=True)
oneline_CC1_all.reset_index(inplace=True)
old_indices = oneline_CC1_all.binary_index
new_indices = np.arange(0,len(old_indices))
map_index = dict(zip(old_indices,new_indices))
oneline_CC1_all.binary_index = new_indices
oneline_CC1_all.set_index('binary_index',inplace=True)  

history_CC1_all.replace({'binary_index':map_index},inplace=True)
history_CC1_all.set_index('binary_index',inplace=True)  


with pd.HDFStore(args.outfile, mode='w') as store:
    store.append('oneline', oneline_CC1_all, data_columns=True)
    store.append('history', history_CC1_all, data_columns=True)