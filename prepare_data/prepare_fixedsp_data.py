import numpy as np
import pandas as pd
from tqdm import tqdm
import os


dss = [
'proton-position-Mom-20358'
,'piminus-position-Mom-39144' 
,'eminus-position-Mom-52294'
#,'kplus-position-Mom-50588'
,'muminus-position-40000'
,'pionzero-position-Mom-41118'
]

n_sp = 500 #max. number of sp in an event

for d in range(len(dss)):
    #ds = 'proton-position-Mom-20358'
    ds = dss[d]
    f_name = './dataset/{:s}.txt'.format(ds)
    evt_folder_path = os.path.join('./dataset/500sp_evts/',ds)

    #evt_dir = os.path.dirname(evt_folder_path)
    if not os.path.isdir(evt_folder_path):
        os.makedirs(evt_folder_path)
        print('Folder created:',evt_folder_path)

    df = pd.read_csv(f_name, sep=' ', names=['evt_num', 'x', 'y', 'z', 'adc', 'sp', 'momentum'])

    evt_columns = ['x','y','z', 'adc']

    n_events = df.tail(1).evt_num.values[0]
    sp_ind_p = 0
    sp_ind = 0
    for i in tqdm(range(n_events)):
        evt_sp = df.sp[sp_ind_p]
        sp_ind = sp_ind+evt_sp
        df_e = df[sp_ind_p:sp_ind][evt_columns] #.to_csv(f_evt_out, index=False)
        
        e_sp = df_e.shape[0]
        if e_sp<n_sp: #interpolate events
            z = np.empty((n_sp,len(evt_columns)))
            z[:] = np.nan
            df_ip = pd.DataFrame(z, columns=evt_columns)

            inds_evt = np.random.permutation(np.arange(n_sp))
            #del_sp = n_sp//e_sp 
            #ind_ip=np.arange(0,n_sp,del_sp) #calculate new indexes for df_ip
            df_ip.iloc[inds_evt[0:e_sp]] = df_e.values
            #df_tosave = df_ip.interpolate(limit=1000, limit_direction='both')
            df_tosave = df_ip.fillna(method='ffill').fillna(method='bfill')
            if df_tosave.isnull().values.any():
                print('Error: missing value in interpolation!')
                print(df_tosave.tail())
        else: #downsample events
            inds_evt = np.random.permutation(e_sp)
            df_tosave = df_e.iloc[inds_evt[0:n_sp]].reset_index(drop=True)
            if df_tosave.isnull().values.any():
                print('Error: missing value in downsampling!')
                print(df_tosave.head())
                print(df_tosave.tail())
            
        
        #pid = np.zeros(n_sp)
        #pid = d
        #df_tosave['pid'] = pid    
        f_evt_out = os.path.join(evt_folder_path, 'evt_{:0>5d}.csv'.format(i))    
        df_tosave.to_csv(f_evt_out, index=False, header=False)
        
        sp_ind_p = sp_ind

        #print('Saved file:', f_evt_out)






