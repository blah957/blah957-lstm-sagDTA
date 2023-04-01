# Resources:

+ README.md: this file.
+ data/davis/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/davis/Y,ligands_can.txt,proteins.txt
  data/kiba/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/kiba/Y,ligands_can.txt,proteins.txt
  These file were downloaded from https://github.com/hkmztrk/DeepDTA/tree/master/data

###  Source codes:
+ csv_data.py: divide the data set in csv format
+ seq_lstm.py: generate lstm-based language features for protein sequences
+ smile_graph.py: converte drug smiles into drug molecule graphs
+ pt_data.py: create data in pytorch format 
+ utils.py: include TestbedDataset used by pt_data.py to create data, and performance measures.
+ layers.py: original definition of the SAGPool layer
+ gat_gcn_sag_g.py: is a network structure proposed in this paper where a single GAT  and a single GCN are connected in series and pooled using the global self-attention graph pooling method
+ gat_gcn_sag_h.py: is a network structure proposed in this paper where a single GAT  and a single GCN are connected in series and pooled using the hierarchical self-attention graph pooling method
+ gcn3_sag_g.py: is a network structure proposed in this paper for comparison experiments, which has 3 GCN layers connected and pooled using  the global self-attention graph pooling method
+ gcn3_sag_h.py:is a network structure proposed in this paper for comparison experiments, which has 3 GCN layers connected and pooled using the hierarchical self-attention graph pooling method
+ training.py: train a lstm_sagDTA model.
+ valid_5_fold.py: validate the model with a five-fold cross-validation format

# Step-by-step running:

## 1. Install Python libraries needed
+ Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
+ Install rdkit: conda install -y -c conda-forge rdkit
+ install SeqVec：pip install seqvec，learn more at  at https://github.com/Rostlab/SeqVec
+ Or run the following commands to install both pytorch_geometric，SeqVec and rdkit:
```
conda create -n LstmSagDTA python=3
conda activate LstmSagDTA
conda install -y -c conda-forge rdkit
conda install pytorch torchvision cudatoolkit -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
pip install seqvec

```

- Or deploy the environment by configuring the yaml file：

```python
conda env create-f LstmSagDTA.yaml
```



## 2. Divide the data set in csv format

Running
```sh
conda activate LstmSagDTA
python csv_data.py
```
 For the benchmark test, each dataset was split into six parts, with one part used for testing and the remaining five parts used for cross-validation and training. This returns davis_fold-1_train.csv，davis_fold-1_valid.csv，... ,davis_fold-5_train.csv,davis_fold-5_valid.csv; davis_test.csv,davis_train.csv; kiba_fold-1_train.csv,kiba_fold-1_valid.csv, ... ,kiba_fold-5_train.csv,kiba_fold-5_valid.csv; and kiba_test.csv,kiba_train.csv,stored at data/processed/.

These files are in turn input to create online data in PyTorch format

## 3. Generate protein language features and drug molecule graphs

Running

```
python seq_lstm.py
```

This returns  seq_lstm.npy, protein language features, stored at 'data/processed/' directory in numpy  format

```
python smile_graph.py
```

This returns  smile_graph.npy, drug molecule graphs features, stored at 'data/processed/' directory in numpy  format

## 4. Train a prediction model

To train a model using training data. The model is chosen if it gains the best MSE for testing data.  
Running 

```sh
python training.py 0 0 0
```

where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively;
 the second argument is for the index of the models, 0/1/2/3 for GAT_GCN_sagg, GAT_GCN_sagh, GCN3_GLOBAL, or GCN3_HIER, respectively;
 and the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively. 
 Note that your actual CUDA name may vary from these, so please change the following code accordingly:

```sh
cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
```

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.
For example, it returns two files train_model_GAT_GCN_sagg_davis_best.pth and train_result_GAT_GCN_sagg_davis.csv when running GAT_GCN_sagg model on Davis data.

## 5. Five-fold cross-validation of the model

In this section, there are four parameters to set, the first three parameters are the same as the settings in section 4. The fourth argument is for the index of the folds, 0/1/2/3/4 for 1 to 5 fold selection respectively.

Train a Five-fold cross-validation model. E.g., running 

```sh
python valid_5_fold.py 0 1 0 0
```

This returns the model achieving the best MSE for validation data throughout the training and performance results of the model on testing data.
For example, it returns two files train_model_GAT_GCN_sagh_davis_fold-1_best.pth and train_result_GAT_GCN_sagh_davis_fold-1.csv when running 1 fold of GAT_GCN_sagh model on Davis data.

