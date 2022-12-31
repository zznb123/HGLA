# HGLA: Mixed High-Order Graph Convolution with Filter Network via LSTM and Channel Attention

A PyTorch implementation of Biomolecular Interaction Prediction based on Mixed High-Order Graph Convolution with Filter Network via LSTM and Channel Attention（HGLA）.

 

------



## Requirements

the codebase is implemented in Python 3.6.13 and the packages used for developments are mentioned below.

```
argparse      1.1
numpy         1.19
torch         1.8.0
torch_sparse  0.6.10
scikit-learn  0.24.2
matplotlib    3.3.4 
scipy         1.5.4
textable      1.6.4
```



------

## Datasets

Four  public biomolecular interaction datasets

BioSNAP-DTI (DTI)   http://snap.stanford.edu/biodata/index.html

BioSNAP-DDI (DDI)  http://snap.stanford.edu/biodata/index.html

HuRI-PPI (PPI)          http://www.interactome-atlas.org/

DisGeNET-GDI (GDI)  https://www.disgenet.org/



## Training options 

Train HGLA with the following command line arguments.

### Input and output options 

```
-network_type  STR  Type of interaction network    Default is "DDI".
--fold_id      INT  Run model on generated fold    Default is 1.
```



### Model options 

```
--seed              INT     Random seed.                   Default is 42.
--epochs            INT     Number of training epochs.     Default is 100.
--batch_size        INT     Number of samples in a batch   Default is 64.
--early-stopping    INT     Early stopping rounds.         Default is 10.
--learning-rate     FLOAT   Adam learning rate.            Default is 5e-4.
--dropout           FLOAT   Dropout rate value.            Default is 0.1.
--order             INT     Order of neighbor including 0  Default is 4.
--dimension         INT     Dimension of each adjacency    Default is 32.
--layers-1          LST     Layer sizes (first).           Default is [32, 32, 32, 32]. 
--layers-2          LST     Layer sizes (second).          Default is [32, 32, 32, 32].
--hidden1           INT     Output of bilinear layer       Default is 64.
--hidden2           INT     Output of last linear layer    Default is 32.
--cuda              BOOL    Run on GPR                     Default is True.
--fastmode          BOOL    Validate every epoch           Default is True
```



------



## Running HGLA for biomolecular interaction prediction 

Train HGLA on DDI network  on fold1

```
python main.py --netowrk_type "DDI" --fold_id 1
```


