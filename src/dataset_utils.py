from glob import glob

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

idx_lookup = {
    'electron-38323':0 #0 eminus-position-Mom-52294
    #,'kplus-position-Mom-50588' #1
    ,'pionminus-39144':1 #1 - kplus yerine bunu kullaniyoruz piminus-position-Mom-39144
    ,'muon-62190':2 #2 muminus-position-40000
    ,'pionzero-35674':3 #3 pionzero-position-Mom-41118
    ,'proton-36793':4 #4 proton-position-Mom-20358
    }



def get_dataset(files, BATCH_SIZE):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    #list_files is not efficient on a large dataset.
    #it takes tens of minutes to start training!
    #ds = tf.data.Dataset.list_files(TRAIN_FILES)
    ds = tf.data.Dataset.from_tensor_slices(files)
    #print('list_files done!')
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    #print('batch done!')
    ds = ds.map(tf_parse_filename, num_parallel_calls=AUTOTUNE)
    #print('map done!')
    ds = ds.cache()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    #print('prefetch done!')

    return ds

def tf_parse_filename(filename):
    """Take batch of filenames and create point cloud and label"""

    def parse_filename(filename_batch):

        pt_clouds = []
        labels = []
        for filename in filename_batch:
            # Read in point cloud
            filename_str = filename.numpy().decode()
            #pt_cloud = np.load(filename_str)
            pt_cloud = np.loadtxt(filename_str, delimiter=',') #, usecols = (0,1,2))

            # Add rotation and jitter to point cloud
            #theta = np.random.random() * 2*3.141
            #A = np.array([[np.cos(theta), -np.sin(theta), 0],
            #              [np.sin(theta), np.cos(theta), 0],
            #              [0, 0, 1]])
            #offsets = np.random.normal(0, 0.02, size=pt_cloud.shape)
            #pt_cloud = np.matmul(pt_cloud, A) + offsets

            # Create classification label
            obj_type = filename_str.split('/')[-2]
            label = np.zeros(5, dtype=np.float32)
            label[idx_lookup[obj_type]] = 1.0

            pt_clouds.append(pt_cloud)
            labels.append(label)

        return np.stack(pt_clouds), np.stack(labels)


    x, y = tf.py_function(parse_filename, [filename], [tf.float32, tf.float32])
    return x, y


def train_val_split(train_size=0.92):
    #train, val = [], []
    #for obj_type in glob('ModelNet40/*/'):
    #    cur_files = glob(obj_type + 'train/*.npy')
    #    cur_train, cur_val = \
    #        train_test_split(cur_files, train_size=train_size, random_state=0, shuffle=True)
    #    train.extend(cur_train)
    #    val.extend(cur_val)
    my_base ='/home/schefke/PIDNet/data/'
    base = '/home/yalmalioglu/dataset5d/500sp_0padding_evts/' #changed by tdps to reflect the new directories
    f_train = my_base+'train_files20k.csv'                  #same as previous comment
    f_test = my_base+'test_files20k.csv'                    #same as previous comment
    df_train = pd.read_csv(f_train, header=None)
    df_test = pd.read_csv(f_test, header=None)
    
    train = base+df_train.iloc[:,1]
    val = base+df_test.iloc[:,1]
    
    #print(train.iloc[0])
    
    return train, val
