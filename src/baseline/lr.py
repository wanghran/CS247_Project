# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:58:00 2019

@author: zhuya
"""
# the code for using logestic regresssion and xgboost
# the code is from the kernel 
# https://www.kaggle.com/dvasyukova/a-linear-model-on-apps-and-labels
"""
some basic result
The result for Logesitic Regression
Current loss 2.280300942132649 accuracy0.1990622278786255
The result for Xgboost
Current loss 2.315592619697098 accuracy0.18381673253399425
"""


import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from utils import load_data

X, y, nclasses = load_data('../data')

def score(X, y, nclasses, model):
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    pred_p = np.zeros((y.shape[0], nclasses))
    for itrain, itest in kf.split(X, y):
        Xtr, Xte = X[itrain, :], X[itest, :]
        ytr = y[itrain]
        model.fit(Xtr, ytr)
        pred_p[itest,:] = model.predict_proba(Xte)
        
    pred = np.argmax(pred_p, axis = -1)
    loss = log_loss(y, pred_p)
    acc = accuracy_score(y, pred)
    print("Current loss {} accuracy{}".format(loss, acc))

print("The result for Logesitic Regression")
lrmodel = LogisticRegression(C=0.02)
score(X, y, nclasses, lrmodel)
print("The result for Xgboost")
xgbmodel = XGBClassifier()
score(X, y, nclasses,xgbmodel)







    
    
    
    
    
    
    
    