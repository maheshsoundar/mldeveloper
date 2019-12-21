# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:33:57 2019

@author: maheshsoundar
"""

import pandas as pd
import numpy as np

#Filters the given dataframe according to criteria. Criteria is specified as a dict with the column name as key
#and condition as well as value to be compared(tuple) as the value of that key. The available conditions that can be
#specified are 'geq,leq,gt,lt,eq,neq'. If filtering based on more than one coliumn is needed, it can just be added to dict as another key,value pair
#in the above mentioned fashion. The filtering of dataframe occurs one after the other( filtered based on first item and resulting df is filtered 
#based on second item and so on)
#Example: filter_dict = {'col1':['eq'],1}. This returns rows of dataframe which had the value of column 'col1' 
#equal to 1. Using other condition and values, results in filtering according to that flag and value. 

class MLFilter():
    def geq(self,df,col,val):
        return df[df[col]>= val]

    def leq(self, df,col,val):
        return df[df[col]<= val]
    
    def gt(self, df,col,val):
        return df[df[col]> val]
    
    def lt(self, df,col,val):
        return df[df[col]< val]
        
    def eq(self,df,col,val):
        return df[df[col]== val]
    
    def neq(self,df,col,val):
        return df[df[col]!= val]
    
    #Parameters: df - Dataframe to be filtered. Should be a dataframe.
    #param- The dict containing the column name as key and condition as well as nueric as the value of that dict. Multiple such items can be added to 
    #to dict and passed into this parameter.
    #Returns: Filtered dataframe
    def do_filter(self,df,param):
        for col,val in param.items():
            try:
                df = getattr(self,val[0].lower())(df,col,val[1])
            except AttributeError:
                print("\n*** Invalid argument given for the type of operation for filter. Valid: geq,leq,gt,lt,eq,neq. Returning df as it is.***\n")
        return df