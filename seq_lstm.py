from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder
import numpy as np
import torch 
import pandas as pd

embedder = SeqVecEmbedder()

total_num_seqs = 0
def lstm_features(seq):
    global total_num_seqs
    total_num_seqs += 1
    print("\rExtract feature for No.{} seq".format(total_num_seqs),end=" ")
    
    embedding = embedder.embed(seq)
    protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L x 1024]
    np_arr = protein_embd.cpu().detach().numpy()
    
    return np_arr

seq_all = []
for dt_name in ['kiba','davis']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('./data/processed/' + dt_name + '_' + opt + '.csv')
        seq_all += list( df['target_sequence'] )
seq_all = set(seq_all)


seq_lstm = {} 
for seq in seq_all:
    l = int(len(seq))
    if l <640:
        T = lstm_features(seq)
        x0 = torch.zeros(640 - l,1024)
        T = torch.cat((torch.tensor(T),x0),0)
    else:       
        T = lstm_features(seq[:640]) # " ".join(list(seq)[0:510]) 
        T = torch.tensor(T)
    
    seq_lstm[seq] = T
    
    # print(seq_lstm[seq].shape,seq_lstm)
    
    
np.save('./data/processed/seq_lstm.npy', seq_lstm) 