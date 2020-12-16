from __future__ import absolute_import
from __future__ import division
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder

#tf.keras.backend.set_floatx('float64')
targets = ['t1','t2','t3','t4','t5']

def preprocess(dataframe):
    
    numeric_columns = dataframe.select_dtypes(['int64','int8','int16']).columns
    dataframe[numeric_columns] = dataframe[numeric_columns].astype('float32')
    
    #ohe = OneHotEncoder()
    dataframe[targets]= pd.get_dummies(dataframe['ratings'])
    
    return dataframe

def standardize(dataframe):
    
    dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
    
    for column, dtype in dtypes:
        if dtype == 'float32':
            dataframe[column] -= dataframe[column].mean()
            dataframe[column] /= dataframe[column].std()
    return dataframe

def load_data():
    
    _cols = ['Review #', 'Brand', 'Variety', 'Style', 'Country', 'ratings']
    
    #train_df = pd.read_csv("files/rtrain_file.csv", names=_cols, dtype='float32')
    #eval_df = pd.read_csv("files/rtest_file.csv", names=_cols, dtype='float32')

    
    train_df = pd.read_csv("gs://testing-traing-jobs-aiplatform/ramen/rtrain_file.csv", names=_cols, dtype='float32')
    eval_df = pd.read_csv("gs://testing-traing-jobs-aiplatform/ramen/rtest_file.csv", names=_cols, dtype='float32')

    train_df = preprocess(train_df)
    eval_df = preprocess(eval_df)

    
    train_y = train_df[targets].copy()
    train_df.drop(targets,inplace=True,axis=1)
    train_x = train_df
    
    eval_y = eval_df[targets].copy()
    eval_df.drop(targets,inplace=True,axis=1)
    eval_x = eval_df

    
    all_x = pd.concat([train_x, eval_x], keys=['train', 'eval'])
    all_x = standardize(all_x)
    train_x, eval_x = all_x.xs('train'), all_x.xs('eval')

    
    #train_y = np.asarray(train_y).astype('float32').reshape((-1, 1))
    #eval_y = np.asarray(eval_y).astype('float32').reshape((-1, 1))

    return train_x, train_y, eval_x, eval_y
