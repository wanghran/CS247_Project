# Graph Embeddings
This directory contains the experiments for DeepWalk and BiNE. 

## DeepWalk
The code is contained in the Jupyter notebook deepwalk.ipynb. To run this code, first download the dataset. Specify the path to the dataset by setting
``` 
datadir = '\path\to\data'
```
Then run the notebook. 

Several pre-calculated graph embedding matrices have also been provided. Each line corresponds to one node. The first entry is the node id, followed by the latent vector.
The file 'data.embeddings' contains the embeddings using a latent dimensionality of 64, and 'data_128.embeddings' uses 128 latent dimensions.

## BiNE
The file to run is Bine.ipynb. This code also requires the dataset to be downloaded. Specify the path to the dataset directory by setting the variable `datadir`.

For ease in running the code, we have included the generated graph embedding matrices in this directory. Each line corresponds to one node. The first entry is the node id, followed by the latent vector.
The file 'vec_u.dat' uses 64 latent dimensions, and 'vec_u_128.dat' uses 128.
