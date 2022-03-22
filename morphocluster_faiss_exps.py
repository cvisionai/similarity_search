import faiss
import pickle
import uuid
import glob
import numpy as np
import os
import pandas as pd

def pd_from_pkl(fname,idx):
    raw_dict = pd.read_pickle(fname)
    columns = ['features_file','fname','label','x1','y1','x2','y2']
    data = {
        'features_file' : fname,
        'fname' : raw_dict.get('fname'),
        'label' : raw_dict.get('label'),
        'x1' : raw_dict.get('roi').get('x1'),
        'y1' : raw_dict.get('roi').get('y1'),
        'x2' : raw_dict.get('roi').get('x2'),
        'y2' : raw_dict.get('roi').get('y2')
    }

    tmp_df = pd.DataFrame(data,columns=columns,index=[idx])

    return tmp_df

def generate_df(pkl_dir):
    flist = glob.glob(os.path.join(pkl_dir,'*.pkl'))
    # This is overkill but should work for arbitrarily large amounts of feature vectors
    indices = [uuid.uuid1().int>>64 for x in range(len(flist))]
    features_df = pd_from_pkl(flist[0],indices[0])
    for fname,idx in zip(flist[1:], indices[1:]):
        features_df = features_df.append(pd_from_pkl(fname,idx))

    return features_df

def load_features(fname):
    return pd.read_pickle(fname).get('features')


if __name__=="__main__":
    features_df = pd.read_pickle('/home/ben/code/fathomnet_utils/fathomnet_features_df.pkl')
    features_files_list = list(features_df.features_file)
    loaded_features = [load_features(f) for f in features_files_list]
    features_np = np.asarray(loaded_features)
    indices = np.asarray(features_df.index,dtype=np.int64)
    index = faiss.IndexFlatL2(960)
    index2 = faiss.IndexIDMap(index)
    index2.add_with_ids(features_np,indices)

    k = 24 # number of nearest results to fetch
    # call example with one of the entries in the database
    D,I = index2.search(features_np[100080:100081],k)
    I_uint = np.asarray(I,dtype=np.uint64) #need to convert back to uint

    # Fetch entries 
    outputs = features_df.loc[I_uint[0]]