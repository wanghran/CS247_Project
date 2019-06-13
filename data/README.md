# Dataset

## the original data set is too large please download from kaggle dataset and put it here, the link is https://www.kaggle.com/c/talkingdata-mobile-user-demographics/data

### Baseline Models
1. 'train_idx.npy' and 'test_idx.npy' contain the index of our train set (80%) and test set (20%) we keep it all the same across all the experiments

2. '*.npz' contains the data after our preprocess. Usually the sparse matrixs that are directly used in our experiments

3. 'active.npz' 'installed.npz' contains the adjacent matrix between user(each line) and APP£¬device(each column). We have 2 matrics because the for each user, there is app installed and app active £¨they share the same device information£©

4. active_wapplabel.npz' contains the original features (concatenation of one-hot vector), it is mainly used in label propagation based GCN model and is concatenated to the feature get from label propagation

5. 'best_model.pth¡¯ is the trained model for label propagated GCN

### hGCN 

The prefix full_ indicates if the file is used as the small dataset or the full size dataset

1. adj_app_active.npz, adj_app_installed.npz and adj_phone.npz are three adjacency matrices, which define the graph.

2. labels.csv contains all the labels in the order of the node id in the graphs. 

3. train_idx and test_idx are the lists of node ids from the train test splt. 

4. full_train_yx and full_test_yx are the lists of device ids of the full size dataset. train_device_id and test_device_id are the lists of device ids of the small dataset. 

5. label_mapping is a mapping that convert the labels codec in the labels.csv or full_labels.csv to the real class names. 
