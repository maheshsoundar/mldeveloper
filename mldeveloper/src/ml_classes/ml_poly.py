# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:28:37 2019

@author: maheshsoundar
"""

import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class MLPolyFeat():

    def __init__(self,poly_deg=2,seed=5):
        self.poly_deg = poly_deg
        self.seed(seed)
        
    def seed(self,seed):
        np.random.seed(seed)
        random.seed(seed)
    
    def fit_transform(self,df):
        return PolynomialFeatures(self.poly_deg).fit_transform(df)