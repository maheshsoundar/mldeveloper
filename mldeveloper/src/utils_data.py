# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:41:00 2019

@author: maheshsoundar
""" 
# Common utility functions. 

import pandas as pd
import random
import numpy as np
 
#Drop cols from given dataframe inplace. Returns the dataframe.
def drop_cols(df,cols):
    if(len(cols)==0):
        print("***\n No column specified to drop ***\n")
        return df
    df.drop(columns=cols,inplace=True)
    return df

#Split the dataframe into features and target and returns both as list of values.
def split_features_targets(df,target):
    if(target not in df.columns.values):
        print("\***Target column specified is not one of the columns***\n")
        return df, None
    y_ = df[target].values
    df.drop(columns=[target],inplace=True)
    return df.values,y_

def get_numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.values

def get_categorical_cols(df):
    numerics = df.select_dtypes(include=np.number).columns.values
    return (set(df.columns.values) - set(numerics)).tolist()

#Shuffle a dataframe. Inputmust be dataframe
def shuffle_df(df,seed=5):
    if(isinstance(df,pd.DataFrame)):
        return df.sample(frac=1,random_state=seed).reset_index(drop=True)
    print("***\n The given input to shuffle_df is not a dataframe. Returning dataframe without shuffling***\n")
    return df

#shuffle an array. Input is a single array. Input must be array
def shuffle_array(arr,seed=5):
    np.random.seed(seed)
    random.seed(seed)
    return np.random.shuffle(arr)

#Shuffle two arrays in unison maintaining the relationship. Input must be array
def shuffle_arr_unison(arr1,arr2,seed=5):
    assert len(arr1) == len(arr2)
    if((isinstance(arr1,pd.DataFrame) == True) or (isinstance(arr1,pd.Series) == True)):
        arr1 = np.array(arr1.values)
    if((isinstance(arr2,pd.DataFrame) == True) or (isinstance(arr2,pd.Series) == True)):
        arr2 = np.array(arr2.values)
        
    random.seed(seed)
    np.random.seed(seed)
    indices = np.arange(arr1.shape[0])
    np.random.shuffle(indices)

    return arr1[indices], arr2[indices]

#Split given dataframes into train and test dataset according to given train_ratio. Inputs must be dataframes or series. 
def train_test_split_df(X_df,Y_df,train_ratio=0.8,shuffle=True,seed=5):
    assert X_df.shape[0]==Y_df.shape[0] 
    np.random.seed(seed)
    random.seed(seed)
    
    X_df.reset_index(drop=True,inplace=True)
    Y_df.reset_index(drop=True,inplace=True)
    n = (X_df.shape[0]*train_ratio).astype(int)
    
    if(shuffle==True):
        train_x = X_df.sample(frac=train_ratio,random_state=seed)
        test_x = X_df.drop(train_x.index)
        train_y = Y_df.iloc[train_x.index]
        test_y = Y_df.iloc[test_x.index]
    else:
        train_x = X_df.iloc[:n,:]
        test_x = X_df.iloc[n:,:]
        train_y = Y_df.iloc[train_x.index]
        test_y = Y_df.iloc[test_x.index]
        
    return train_x, test_x, train_y, test_y

#Split given arrays into train and test dataset according to given train_ratio.Inputs must be arrays.
def train_test_split_arr(X,Y,train_ratio=0.8,shuffle=True,seed=5):
    assert len(X)==len(Y) 
    np.random.seed(seed)
    random.seed(seed)

    n = int(len(X)*train_ratio)
    if(shuffle==True):
        X, Y = shuffle_arr_unison(X,Y,seed=seed)
        
    return X[:n], X[n:], Y[:n], Y[n:]
 
def get_not_unique_cols(df,threshold=1):
    if(isinstance(df,pd.DataFrame) == False):
       print("***\n The given input is not a dataframe. Returning dataframe without removing any columns***\n")
       return df
    n_uniq = df.nunique()
    remove_cols = n_uniq[n_uniq<=threshold].index.values
    return remove_cols

def get_multicollinear_cols(df,threshold=0.98):
    if(isinstance(df,pd.DataFrame) == False):
       print("***\n The given input is not a dataframe. Returning dataframe without removing any columns***\n")
       return df
   
    numerics = get_numeric_cols(df)
    corr_matrix = df[numerics].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    remove_cols = [column for column in upper.columns if any(upper[column] >= threshold)]  
    return remove_cols

def get_no_std_cols(df,threshold=0.1):
    if(isinstance(df,pd.DataFrame) == False):
       print("***\n The given input is not a dataframe. Returning dataframe without removing any columns***\n")
       return df
   
    numerics = get_numeric_cols(df)
    std = df[numerics].std()
    remove_cols = std[std <= threshold].index.values
    return remove_cols