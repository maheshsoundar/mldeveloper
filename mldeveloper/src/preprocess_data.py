# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:42:45 2019

@author: maheshsoundar
"""

import pandas as pd
import random
import numpy as np
from .utils_data import get_multicollinear_cols,get_no_std_cols,get_not_unique_cols

# Pre-process the data. Functions like one-hot encoding, removing unnecessary features, dropping rows containing missing values are done.
#To be intialized with seed. The object created can be used to call standard_preprocess to process the data.
class MLPreprocess():
    def __init__(self,seed=5):
        self.seed(seed)
        self.selected_columns = []
    
    def clear(self):
        self.selected_columns = []
        
    def seed(self,seed):
        np.random.seed(seed)
        random.seed(seed)
    
    def encode_categorical_onehot(df,columns=None):
        if(isinstance(df,pd.DataFrame) == False):
           print("***\n The given input is not a dataframe. Returning as it is***\n")
           return df
       
        return pd.get_dummies(df,columns=columns,drop_first=True)

    def remove_multicollinear_cols(self,df,threshold=0.98):
        if(isinstance(df,pd.DataFrame) == False):
           print("***\n The given input is not a dataframe. Returning as it is***\n")
           return df
       
        remove_cols = get_multicollinear_cols(df)
        df.drop(df[remove_cols],axis=1,inplace=True)
        return df
    
    def remove_no_std_cols(self,df,threshold=0.1):
        if(isinstance(df,pd.DataFrame) == False):
           print("***\n The given input is not a dataframe. Returning as it is***\n")
           return df
       
        remove_cols = get_no_std_cols(df)
        df.drop(df[remove_cols],axis=1,inplace=True)
        return df
    
    def remove_not_unique_cols(self,df,threshold=0.1):
        if(isinstance(df,pd.DataFrame) == False):
            print("***\n The given input is not a dataframe. Returning as it is***\n")
            return df
       
        remove_cols = get_not_unique_cols(df)
        df.drop(df[remove_cols],axis=1,inplace=True)
        return df
        
    def standard_preprocess_intern(self,df,unique_thresh=1,multicoll_thresh=0.98,std_thresh=0.1,encode_cols=None,encode=False,dropna=False,remove_cols_analysis=False):
        if(isinstance(df,pd.DataFrame) == False):
           print("***\n The given input is not a dataframe. Returning as it is***\n")
           return df
       
        if(dropna):
            df.dropna(inplace=True)
        
        if(remove_cols_analysis):
            if(len(self.selected_columns)==0):
                df = self.remove_not_unique_cols(df,unique_thresh)
                df = self.remove_no_std_cols(df,std_thresh)
                df = self.remove_multicollinear_cols(df,multicoll_thresh)
                self.selected_columns = df.columns.values
            else:
                df = df[self.selected_columns]
            
        if(encode):
            df = self.encode_categorical_onehot(df,encode_cols)
            
        return df
     
    #Parameters:
    #list_dfs - list containing the dataframes that needs pre-processing. If only one dataframe needs to be pre-processed, then list could contain 
    #only one dataframe.
    #remove_cols_analysis - default False. If True, features with less than 'n' unique values in them, multi-collinear features ábove given threshold and
    #features with standard deviation less than given threshold will be removed. Thresholds must be specified seperately using other parameters.
    #unique_thresh - default 1. Features with no of unique values less than or equal to threshold is removed.
    #multicoll_thresh - default 0.98. Degree of multi-collinearity allowed, above which one of the featurs is removed.
    #std_thresh - default 0.1 - Minimum standard deviation in each column allowed, below which removed
    #encode - default False. Flag to one-hot-encode categorical features or not. If ture and encode_cols not provided, columns tpyes other than int and float,
    #automatically encoded.
    #encode_cols - default None. list of column names to encode.
    #dropna - default False. Falg to drop rows with missing values.
    
    #returns: 
    #new_list_dfs - list containing pre-processed dataframes, available in the order they were given in as input.
    def standard_preprocess(self,list_dfs,unique_thresh=1,multicoll_thresh=0.98,std_thresh=0.1,encode_cols=None,encode=False,dropna=False,remove_cols_analysis=False):
        new_list_dfs = []
        for df in list_dfs:
            assert df is not None
            new_list_dfs.append(self.standard_preprocess_intern(df, unique_thresh=unique_thresh,multicoll_thresh=multicoll_thresh,std_thresh=std_thresh,encode_cols=encode_cols,encode=encode,
                                                                dropna=dropna,remove_cols_analysis=remove_cols_analysis))
        self.clear()
        return new_list_dfs
        