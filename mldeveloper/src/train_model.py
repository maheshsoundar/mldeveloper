# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 23:47:24 2019

@author: MAASORI
"""

import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score,accuracy_score,confusion_matrix

#Train the model using an object created using this class. Call train_test using the object to train and get score for the model.
#Parameters:
#estimator - default 'linreg'. Type of algorithm to be used to create the model. Available types 'linreg' for linear regression, 'logreg' for logistic regression,
#'rf_reg' for random forest regressor, 'rf_class' for random forest classifier.
#type_problem - default 'regression'. Type of problem. Available options are 'regression' or 'classification'.
#n_trees - default 10. No of trees to build if random forest model is selected.
#seed - default 5. For reproducibility
class MLModel():

    def __init__(self,estimator='linreg', type_problem='regression', n_trees=10, seed=5): 
        self.seed(seed)
        self.estimator = estimator
        self.type_problem = type_problem
        self.n_trees = n_trees
        
    def seed(self,seed):
        np.random.seed(seed)
        random.seed(seed)
        self.nseed = seed
     
    def linreg(self,x,y):
        print("*** Training with linear regression ***\n")
        return LinearRegression().fit(x,y)
    
    def logreg(self,x,y):
        print("*** Training with logistic regression ***\n")
        return LogisticRegression(verbose=1, random_state=self.nseed).fit(x,y)
    
    def rf_class(self,x,y):
        print("*** Training with random forest classifier ***\n")
        return RandomForestClassifier(n_estimators=self.n_trees, verbose=1, random_state=self.nseed).fit(x,y)
    
    def rf_reg(self,x,y):
        print("*** Training with random forest regressor ***\n")
        return RandomForestRegressor(n_estimators=self.n_trees, verbose=1, random_state=self.nseed).fit(x,y)
    
    #Returns: #estimator - estimator trained
    #mae - Mean absolute error of trained model on test data in case of 'regression'
    #r2score - r2score of trained model on test data in case of 'regression'
    #accuracy - accuracy of trained model on test data in case of 'classification'
    #confusion_matrix - confusion matrix of trained model on test data in case of 'classification'
    def train_test(self,x_train,y_train,x_test,y_test):
        assert len(x_train)==len(y_train)
        assert len(x_test)==len(y_test)
        
        estimator = self.do_train(x_train,y_train)
        print(estimator)
        y_pred = estimator.predict(x_test)
        if(self.type_problem == 'regression'):
            return estimator, mean_absolute_error(y_test,y_pred), r2_score(y_test,y_pred)
        else:
            return estimator, accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred)
        
    def do_train(self,x,y):
        estimator = None
        try:
            estimator = getattr(self,self.estimator.lower())(x,y)
        except AttributeError:
            print("\n*** Invalid argument given for the type of estimator. Valid: linreg,logreg,rf_class,rf_reg. Returning None.***\n")
        return estimator
    