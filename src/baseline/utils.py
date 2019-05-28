# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:29:30 2019

@author: zhuya
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

def load_data(datadir = '../data'): 
    # load in all the data
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
    # feature engineering for using 
        # phone brand
        # device model
        # installed apps
        # app labels
        
    #phone band and device model
    gatrain['trainrow'] = np.arange(gatrain.shape[0])
    brandencoder = LabelEncoder().fit(phone.phone_brand)
    phone['brand'] = brandencoder.transform(phone['phone_brand'])
    gatrain['brand'] = phone['brand']
    # this matrix represent the original brand in a sparse matrix 
    # every row correspond to 1 person, represent by 1-hot vector indicate the brand
    Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                           (gatrain.trainrow, gatrain.brand)))
    print('Brand features: train shape {}'.format(Xtr_brand.shape))
    
    m = phone.phone_brand.str.cat(phone.device_model)
    modelencoder = LabelEncoder().fit(m)
    phone['model'] = modelencoder.transform(m)
    gatrain['model'] = phone['model']
    Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                           (gatrain.trainrow, gatrain.model)))
    print('Model features: train shape {}'.format(Xtr_model.shape))
    
    #get app information --> what app is installed
    appencoder = LabelEncoder().fit(appevents.app_id)
    appevents['app'] = appencoder.transform(appevents.app_id)
    napps = len(appencoder.classes_)
    deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                           .groupby(['device_id','app'])['app'].agg(['size'])
                           .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                           .reset_index())
    
    d = deviceapps.dropna(subset=['trainrow'])
    Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                          shape=(gatrain.shape[0],napps))
    print('Apps data: train shape {}'.format(Xtr_app.shape))
    
    #get app label data --> the label for installed app --> e.g. game, chat, etc.
    applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
    applabels['app'] = appencoder.transform(applabels.app_id)
    labelencoder = LabelEncoder().fit(applabels.label_id)
    applabels['label'] = labelencoder.transform(applabels.label_id)
    nlabels = len(labelencoder.classes_)
    devicelabels = (deviceapps[['device_id','app']]
                    .merge(applabels[['app','label']])
                    .groupby(['device_id','label'])['app'].agg(['size'])
                    .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                    .reset_index())
    devicelabels.head()
    
    d = devicelabels.dropna(subset=['trainrow'])
    Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                          shape=(gatrain.shape[0],nlabels))
    print('App labels data: train shape {}'.format(Xtr_label.shape))
    X = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')
    print('All features: train shape {}'.format(X.shape))
    
    targetencoder = LabelEncoder().fit(gatrain.group)
    y = targetencoder.transform(gatrain.group)
    nclasses = len(targetencoder.classes_)
    return X, y, nclasses

def load_data_only_brand_app(datadir = '../data'): 
    gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                              index_col='device_id')
    phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
    phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
    events = pd.read_csv(os.path.join(datadir,'events.csv'),
                         parse_dates=['timestamp'], index_col='event_id')
    appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                            usecols=['event_id','app_id','is_active'],
                            dtype={'is_active':bool})
    applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
    gatrain['trainrow'] = np.arange(gatrain.shape[0])
    m = phone.phone_brand.str.cat(phone.device_model)
    modelencoder = LabelEncoder().fit(m)
    phone['model'] = modelencoder.transform(m)
    gatrain['model'] = phone['model']
    #get device model
    Xtr_model = np.array(gatrain.model, dtype=np.int)
    print('Model features: train shape {}'.format(Xtr_model.shape))
    
    #get event list
    appencoder = LabelEncoder().fit(appevents.app_id)
    appevents['app'] = appencoder.transform(appevents.app_id)
    napps = len(appencoder.classes_)
    deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                           .groupby(['device_id','app'])['app'].agg(['size'])
                           .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                           .reset_index())
    d = deviceapps.dropna(subset=['trainrow'])
    Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                          shape=(gatrain.shape[0],napps)) 
    print('App features: train shape {}'.format(Xtr_app.shape))
    
    #get app label data --> the label for installed app --> e.g. game, chat, etc.
    applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
    applabels['app'] = appencoder.transform(applabels.app_id)
    labelencoder = LabelEncoder().fit(applabels.label_id)
    applabels['label'] = labelencoder.transform(applabels.label_id)
    nlabels = len(labelencoder.classes_)
    devicelabels = (deviceapps[['device_id','app']]
                    .merge(applabels[['app','label']])
                    .groupby(['device_id','label'])['app'].agg(['size'])
                    .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                    .reset_index())
    devicelabels.head()
    
    d = devicelabels.dropna(subset=['trainrow'])
    Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                          shape=(gatrain.shape[0],nlabels))
    

    print('App features: train shape {}'.format(Xtr_label.shape))
    #get label
    targetencoder = LabelEncoder().fit(gatrain.group)
    y = targetencoder.transform(gatrain.group)
    print('Label shape {}'.format(y.shape))
    nclasses = len(targetencoder.classes_)
    return Xtr_model, Xtr_app, y, nclasses
    
















