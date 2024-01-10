import pandas as pd
import numpy as np
from utils import *

smile_graph = np.load('data/processed/smile_graph.npy',allow_pickle=True).item()
seq_lstm = np.load('data/processed/seq_lstm.npy',allow_pickle=True).item()



def create_data(dataset):
    df = pd.read_csv('data/processed/' + dataset + '_train.csv')
    train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
    train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
    train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph,seq_lstm=seq_lstm)
    
    df = pd.read_csv('data/processed/' + dataset + '_test.csv') #加了fold  test改成了valid
    test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
    test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, xt=test_prots, y=test_Y,
    smile_graph=smile_graph,seq_lstm=seq_lstm)
    
    return train_data,test_data

def create_data_5_fold(dataset,fold):

    df = pd.read_csv('data/processed/' + dataset + '_' + 'fold-' + str(fold) + '_train.csv')
    train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
    train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
    train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph,seq_lstm=seq_lstm)
    
    df = pd.read_csv('data/processed/' + dataset + '_' + 'fold-' + str(fold) + '_valid.csv') #加了fold  test改成了valid
    test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
    test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, xt=test_prots, y=test_Y,
    smile_graph=smile_graph,seq_lstm=seq_lstm)
    
    return train_data,test_data
       
