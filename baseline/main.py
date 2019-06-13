# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:58:00 2019

@author: zhuya
"""
# the code for using logestic regresssion and xgboost
# the code is from the kernel 
# https://www.kaggle.com/dvasyukova/a-linear-model-on-apps-and-labels

import gc
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from utils import load_data, load_data_seperate_label
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, filename = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=6.5)
    fig.tight_layout()
    if filename:
        fig.savefig(filename + '.png')
    return ax


def fit_classification(X, y, class_name, model, filename=None, top3=True):
    itrain = np.load('../data/train_idx.npy')
    itest = np.load('../data/test_idx.npy')
    Xtr, Xte = X[itrain, :], X[itest, :]
    ytr = y[itrain]
    yte = y[itest]
    model.fit(Xtr, ytr)
    pred_p = model.predict_proba(Xte)
    pred = np.argmax(pred_p, axis = -1)
    loss = log_loss(yte, pred_p)
    acc1 = accuracy_score(yte, pred)
    plot_confusion_matrix(yte, pred, class_name, True, None, plt.cm.Blues, filename)
    if top3 == True:
        acc3 = 0
        top3_pred = np.argsort(pred_p, axis =-1)[:,-3:]
        numdata = len(top3_pred)
        for i in range(numdata):
            cury = yte[i]
            curpred = top3_pred[i]
            if cury in curpred:
                acc3 += 1
        acc3 /= numdata   
        print("Current loss {} top1 accuracy{} top3 accuracy{}".format(loss, acc1, acc3))
    else:
        print("Current loss {} top1 accuracy{}".format(loss, acc1))
def fit_regression(X, y, model, filename=None):
    itrain = np.load('../data/train_idx.npy')
    itest = np.load('../data/test_idx.npy')
    Xtr, Xte = X[itrain, :], X[itest, :]
    ytr = y[itrain]
    yte = y[itest]
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    pred = np.clip(pred, a_min=0, a_max=100)
    meandiff = np.mean(np.abs(yte - pred))
    pr, _ = stats.pearsonr(yte, pred)
    print("Average predicted age difference is {}, pearsonr r is {} ".format(meandiff, pr))
    plt.figure()
    plt.scatter(yte, pred, s=10)
    plt.savefig(filename)
        
mode = 'separate'
if mode == 'together':
    X, y, class_name = load_data('../data')

    print("The result for Logesitic Regression")
    lrmodel = LogisticRegression(C=0.02)
    fit_classification(X, y, class_name, lrmodel, 'cm_lr')

    print("The result for Xgboost")
    xgbmodel = XGBClassifier()
    fit_classification(X, y, class_name, xgbmodel, 'cm_xgb')


elif mode == 'separate':
    
    X, gender, class_name, age = load_data_seperate_label('../data') 
    print("The result for gender model")
    gendermodel = LogisticRegression(C=0.02)
    fit_classification(X, gender, class_name, gendermodel, filename='gender', top3=False)
    print("The result for age model")
    gc.collect()
    agemodel = LinearRegression()
    fit_regression(X, age, agemodel, 'cm_age')
    
    
else:
    pass










    
    
    
    
    
    
    
    