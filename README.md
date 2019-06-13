# User profile prediction in heterogeneous information networks

We want to solve a problem from an old Kaggle competition which can be found [here](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/data). It requires to predict user profile based on the information of app usage, phone type, geolocation and timestamp when data is collected.  

## Data Pre-processing

Based on the data from Kaggle, we pre-process it and obtain two version of datasets:
* A subset of the original dataset which only contains users that at least have app information. This data set is consideredto be less noisy and contain more information.
* The full data set which contains all the data with and without app information. Since app information is very important for this classification, this full set contains more noise and thus is considered to be much harder.

The files with the prefix of 'full' in the folder named data correspond to the information extracted from the full dataset, whereas files without that prefix are relevant to small dataset.

## Models

We take three approaches in total:
* Baseline Models (Logistic Regression; Gradient Boosting)
* Traditional Graph Embedding Models (DeepWalk; BiNE)
* Graph Convolutional Neural Network (Heterogeneous GCN; Label-Propagation based GCN)

### Baseline Models

The folder named baseline contains all the files about baseline models.

### Traditional Graph Embedding Models

The folder named graph_embedding consists of all the files about Traditional Graph Embedding Models including DeepWalk and BiNE.

### Graph Convolutional Neural Network

The folder called hGCN contains all the work about heterogeneous GCN, while the folder named label_prop_GCN includes files about Label-Propagation based GCN. 


## Authors

* **Haoran Wang** 
* **Megan Williams** 
* **Haiqi Xiao** 
* **Yaxuan Zhu** 