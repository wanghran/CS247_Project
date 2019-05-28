# -*- coding: utf-8 -*-
"""
Created on Fri May 17 19:59:10 2019

@author: zhuya
"""
import numpy as np
import time
from sklearn.metrics import log_loss, accuracy_score
from utils import load_data_only_brand_app

'''
Very naive method, predict the label of a given person as the average of those people
he connect to in the graph
Current loss 3.4605982619697757 accuracy0.15862808145766344
'''

X_model, X_app, y, nclasses = load_data_only_brand_app()
num_data = X_app.shape[0]
num_test = int(0.1 * num_data)
full_idx = np.arange(num_data)
np.random.shuffle(full_idx) 
train_idx = full_idx[num_test:]
test_idx = full_idx[:num_test]

Xtr_model = X_model[train_idx]
Xtr_app = X_app[train_idx].transpose()
'''
print(Xtr_app[0,:])
rows, cows = Xtr_app[0,:].nonzero()
for i in range(len(rows)):
    print(rows[i], cows[i])
'''
ytr = y[train_idx]
Xte_model = X_model[test_idx]
Xte_app = X_app[test_idx]
yte = y[test_idx]

prob = np.zeros((num_test, nclasses))
model_dic = {}
for i in range(len(Xtr_model)):
    if Xtr_model[i] not in model_dic.keys():
        model_dic[Xtr_model[i]] = [i]
    else:
        model_dic[Xtr_model[i]].append(i)

start_time = time.time()
# begin testing
for i in range(num_test):
    tmp_prob = np.zeros((nclasses))
    counter = 0
    _, c = Xte_app[i, :].nonzero()
    # get all the app this person installed
    for app in c:
        _, p_list = Xtr_app[app, :].nonzero()
        for p in p_list:
            counter += 1
            tmp_prob[ytr[p]] += 1.0
    # get the people with same phone model    
    cur_device_model = Xte_model[i]
    if cur_device_model in model_dic.keys():
        p_list = model_dic[cur_device_model]
        for p in p_list:
            counter += 1
            tmp_prob[ytr[p]] += 1.0

    
    
    if counter == 0:
        prob[i] = 1.0/nclasses * np.ones((nclasses))
    else:
        tmp_prob /= counter
        prob[i] = tmp_prob
    if i % 200 == 0:
        print("Time {}, i {}".format(time.time()- start_time, i))

pred = np.argmax(prob, axis = -1)
loss = log_loss(yte[:num_test], prob[:num_test])
acc = accuracy_score(yte[:num_test], pred[:num_test])
print("Current loss {} accuracy{}".format(loss, acc))
    
    
    
    
