This directory contains our data

** the original data set is too large please download from kaggle dataset and put it here

** the link is https://www.kaggle.com/c/talkingdata-mobile-user-demographics/data

** 'train_idx.npy' and 'test_idx.npy' contain the index of our train set (80%) and test set (20%) we keep it all the same across all the experiments

** '*.npz' contains the data after our preprocess. Usually the sparse matrixs that are directly used in our experiments

** 'active.npz' 'installed.npz' contains the adjacent matrix between user(each line) and APP£¬device(each column). We have 2 matrics because the for each user, there is app installed and app active £¨they share the same device information£©

** 'active_wapplabel.npz' contains the original features (concatenation of one-hot vector), it is mainly used in label propagation based GCN model and is concatenated to the feature get from label propagation

** 'best_model.pth¡¯ is the trained model for label propagated GCN

