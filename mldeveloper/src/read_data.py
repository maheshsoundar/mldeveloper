# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:42:45 2019

@author: maheshsoundar
"""

import pandas as pd
from os import listdir
from os.path import isfile, join
import random
import numpy as np

#Reads the files available in given filepath. Creates seperate train, test, validation dataframes if needed. Filepath must be provided. Other
#params can be ignored if not needed.

#Parameters:
#filepath - string specifying the folder in which all the file to be read are present.
#testfile_list - list of strings specifying the names of the files in the folder which must considered as testing data. Seperate datarame is created
#for these and returned.
#valfile_list - list of strings specifying the names of the files in the folder which must considered as validation data. Seperate datarame is created
#for these and returned.
#col_train - List containing names of columns that must be read from the file.
#col_lag - Dict containing names of columns and its correspondin lag numeric as value of that key. The given column is considered and new column with 
#lagged values of the specified column is created. The name of lagge column ends with '(t-1)'. The amount of lag depends on provided value to that 
#particular column. Example: col_lag = {"col1":1}. Here a new column with name "col1(t-1)" is created and filled with lagged values of "col1" 
#with timestep 1.
#seed- seed for reproducibility.
class MLReader():
    def __init__(self,filepath,testfile_list=[],valfile_list=[],col_train=[],col_lag=dict(),seed=5):
        assert filepath is not None
        
        self.path = filepath
        self.train_cols = col_train
        self.lag_cols = col_lag
        self.val_files = valfile_list
        self.test_files = testfile_list
        self.train_files = []
        self.files = []
        self.seed(seed)
        
    def seed(self,seed):
        np.random.seed(seed)
        random.seed(seed)
    
    #scans the given folder to create a list of files with required extension that are present in the folder. These files will be read into dataframe.    
    def scan_files(self,extension='.txt'):
        if not self.path:
            print("\n*** Total no of files scanned from folder with extension {}: {} ***\n".format(extension,len(self.files)))
            return self.files
        
        for f in listdir(self.path):
            if (isfile(join(self.path, f))):
                if(f.endswith(extension)):
                    self.files.append(f)
                    
        print("\n*** Total no of files scanned from folder with extension {}: {} ***\n".format(extension,len(self.files)))
        return self.files
    
    #seperate train files from test and validation file, if test or validation file list is provided.
    def get_train_files(self):
        if(len(self.files)==0):
            print("\n*** No files to read from. Source file list empty ***\n")
            return []
        
        self.train_files = list(set(self.files) - set(self.val_files))
        self.train_files = list(set(self.train_files) - set(self.test_files))     
        return self.train_files
    
    #creates the lagged columns
    def create_lag_cols(self,df,dropnan=True):
        if(len(self.lag_cols)==0):
            return df
        
        for col,n_in in self.lag_cols.items():
            for i in range(n_in, 0, -1):
                col_name = col + "(t-" + str(i) + ')'
                df[col_name] = df[col].shift(i)
                
        # drop rows with NaN values
        if dropnan:
            df.dropna(inplace=True)
        return df

    #read given list of files onto a dataframe
    def read_files_df(self,list_files):
        if not list_files:
            return None
        
        list_df = []
        counter = 1
        for file in list_files:
            df = pd.read_csv(self.path+"\\"+file, delim_whitespace=True,usecols=self.train_cols)
            df = self.create_lag_cols(df)
            print("File {} with name {} read".format(counter,file))
            list_df.append(df) 
            counter+=1
        print("\n")    
        res_df = pd.concat(list_df, axis=0, ignore_index=True)
        return res_df
    
    #read files seperately into train, test and validation dataframes as specified by seperate list of file names for test and/or validation.
    #Only needed if they need to be seperately provided
    #Returns: train , test and validation dataframes.
    def get_all_df(self):       
        self.scan_files()
        self.get_train_files()
        train_df = self.read_files_df(self.train_files)
        test_df = self.read_files_df(self.test_files)
        val_df = self.read_files_df(self.val_files)
        
        return train_df,test_df,val_df
    
    #Read all the files which satisfy the extension in the given filepath into a dataframe.
    #Returns: the dataframe with read content. 
    def get_all_data(self):       
        self.scan_files()
        return self.read_files_df(self.files)
    


