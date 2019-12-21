# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:28:37 2019

@author: maheshsoundar
"""

import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#class to scale data. Provide chunksize and type of scaler. Use the object of created class to call scale method with dataframe as input.

class MLScaler():

    def __init__(self,scaler='minmax',chunk_size=1000,seed=5):
        if(scaler == 'std'):
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        self.chunksize = chunk_size
        self.seed(seed)
        
    def seed(self,seed):
        np.random.seed(seed)
        random.seed(seed)
      
    def partial_fit(self,df):
        chunks = np.array_split(df,df.shape[0]/min(self.chunksize,df.shape[0]))
        for chunk in chunks:
            self.scaler.partial_fit(chunk)
        return self.scaler
    
    def transform(self,df):
        return np.array(self.scaler.transform(df))
    
    def scale(self,df):
        self.partial_fit(df)
        return self.transform(df), self.scaler
          
          
	