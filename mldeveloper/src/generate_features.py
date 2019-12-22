# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:55:28 2019

@author: maheshsoundar
"""

import pandas as pd
import random
import numpy as np
from sklearn.decomposition import PCA
from .ml_classes.ml_poly import MLPolyFeat
from .ml_classes.ml_scaler import MLScaler
from .utils_data import train_test_split_arr

#Generate features before training. Initialize the object of the class with necessary parameters and call process_for_training.
#shuffle: defalut True. If False, no shuffling takes place during train test split.
#poly: default False. Flag to specify if polynomial fetures needs to calculated.
#poly_deg: default 2: Degree of polynomial features to be calculated. Uses scikit-learn. 
#scale: default False: Flag to specify if scaling is needed.
#scaler: default minmax(Min Max Scaling). Type of scaling. Other option is 'std' which is standard scaling. Uses scikit-learn.
#chunksize: default 1000. Chunk size for partial fit of scaler. Useful when dealing with large amounts of data and less memory.
#pca: default False: Flag to specify if Principal Component Analysis is need. Uses scikit-learn.
#n_components: default 0.98. Ratio of cumulative variance to be considered when taking the top n components using pca.
#train_ratio: default=0.8. Ratio of training data needed when splitting into train and test sets. Remaining data will be test data.
class MLFeatureGenerate():
    def __init__(self,seed=5,shuffle=True,poly=False,poly_deg=2,scale=False,scaler='minmax',chunksize=1000,pca=False,n_components=0.98,train_ratio=0.8):
        self.seed(seed)
        self.shuffle = shuffle
        self.poly = poly
        self.poly_deg = poly_deg
        self.scale = scale
        self.scaler = scaler
        self.chunksize = chunksize
        self.pca = pca
        self.n_components = n_components
        self.train_ratio = train_ratio
        
    def seed(self,seed):
        np.random.seed(seed)
        random.seed(seed)
        self.nseed = seed
    
    def create_poly_feats(self,x):
        poly_feat = MLPolyFeat(self.poly_deg,self.nseed)
        x = poly_feat.fit_transform(x)
        return x, poly_feat
    
    def scale_feats(self,x):
        scaler_feat = MLScaler()
        x, scaler_feat = scaler_feat.scale(x)
        return x, scaler_feat
    
    def pca_feat(self,x):
        pca_feat = PCA(n_components=self.n_components,random_state=self.nseed).fit(x)
        return pca_feat.transform(x), pca_feat
    
    #Returns: #x_train - Resulting train data fetures
    #x_test - Test data features
    #y_train - Resulting train data target
    #y_test - Test data target
    #poly_feat - Object with which poly feats were created. None if not used.
    #scaler_feat - Object with which scaling was done.  None if not used.
    #pca_feat  - Object with which pca was done. None if not used.
    def process_for_training(self,x,y):
        poly_feat = None
        scaler_feat = None
        pca_feat = None
        
        if((isinstance(x,pd.DataFrame) == True) or (isinstance(x,pd.Series) == True)):
            x = np.array(x.values)
        if((isinstance(y,pd.DataFrame) == True) or (isinstance(y,pd.Series) == True)):
            y = np.array(y.values)
            
        if(self.poly==True):
            x, poly_feat = self.create_poly_feats(x)
        if(self.scale==True):
            x, scaler_feat = self.scale_feats(x)
        if(self.pca==True):
            x, pca_feat = self.pca_feat(x)
        
        x_train, x_test, y_train, y_test = train_test_split_arr(x, y, train_ratio=self.train_ratio, seed=self.nseed, shuffle=self.shuffle)
        return x_train, x_test, y_train, y_test, poly_feat, scaler_feat, pca_feat
    
