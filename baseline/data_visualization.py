# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:15:30 2019

@author: zhuya
"""
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

def plot_bar(height_in, label_in, file_name, rotate = False, fontsize=10, w=0.6, addothers=True):
    # deal with the encoding problem of chinese character
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    if len(height_in) > 20:
        height = []
        label = []
        idx = np.argsort(height_in)
        for i in range(1, 20):
            print(height_in[idx[-i]])
            print(label_in[idx[-i]])
            height.append(height_in[idx[-i]])
            label.append(label_in[idx[-i]])
        if addothers: 
            height.append(np.sum(height_in) - np.sum(height))
            label.append('others')
        else:
            height.append(height_in[idx[-20]])
            label.append(label_in[idx[-20]])
    else:
        height = height_in
        label = label_in
    x = np.arange(len(height))
    fig, ax = plt.subplots()
    plt.bar(x, height, width=w)
    plt.xticks(x, label, fontsize=fontsize)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.savefig(file_name)
def plot_histogram(data, bins, file_name):
    fig, ax = plt.subplots()
    plt.hist(data, bins)
    fig.savefig(file_name)


part_data = False
if part_data:
    train_idx = np.load('train_idx.npy')
    test_idx = np.load('test_idx.npy')
    all_idx = np.concatenate([train_idx, test_idx])


datadir = '../data'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                          index_col='device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),
                         parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                            usecols=['event_id','app_id','is_active'],
                            dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))

age = np.array(gatrain['age'])
gender = np.array(gatrain['gender'])
if part_data == True:
    age = age[test_idx]
    gender = gender[test_idx]


M_F = [np.sum(gender == 'M')/len(gender), np.sum(gender == 'F')/len(gender)]


plot_bar(M_F, ['Male', 'Female'], 'gender.png', False, 15, 0.4) 
plot_histogram(age, 30, 'age.png')
age_gender = np.array(gatrain['group'])
groups = np.unique(age_gender)
if part_data == True:
    age_gender = age_gender[test_idx]    
count = []
for i in range(len(groups)):
    count.append(np.sum(age_gender == groups[i]) / len(age_gender))
plot_bar(count, groups, 'groups.png', True, 9, 0.8)

brand = np.array(phone['phone_brand'])
brand_name = np.unique(brand)
count = []
for i in range(len(brand_name)):
    count.append(np.sum(brand == brand_name[i]) / len(brand))

plot_bar(count, brand_name, 'brand.png', True, 8, 0.8)

model = np.array(phone.phone_brand.str.cat(phone.device_model))
model_name = np.unique(model)
count = []
for i in range(len(model_name)):
    count.append(np.sum(model==model_name[i])/len(model))
plot_bar(count, model_name, 'model.png', True, 6, 0.8, False)

AppCat = np.array(applabels['label_id'])
Cat_id = np.unique(AppCat)
count = []
Cat_name = []
label_cat = pd.read_csv(os.path.join(datadir,'label_categories.csv'), index_col='label_id')  

for i in range(len(Cat_id)):
    count.append(np.sum(AppCat==Cat_id[i])/len(AppCat))
    Cat_name.append(label_cat.loc[Cat_id[i]].category)    
   
plot_bar(count, Cat_name, 'appcategory.png', True, 6, 0.8, False)
 
    
    