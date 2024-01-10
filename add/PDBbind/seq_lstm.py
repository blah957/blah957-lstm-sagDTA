#from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder
# from bio_embeddings.embed import SeqVecEmbedder
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import torch 
import pandas as pd
import os

weights_path='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
options_path='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
root_dir='./seqVec'
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

weights = os.path.join(root_dir, 'weights.hdf5')
options = os.path.join(root_dir, 'options.json')

def download_file(url, filename):
  response = requests.get(url, stream=True)
  with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                    total=int(response.headers.get('content-length', 0)),
                    desc=filename) as fout:
      for chunk in response.iter_content(chunk_size=4096):
          fout.write(chunk)

if not os.path.exists(weights):
    download_file(weights_path, weights)
if not os.path.exists(options):
    download_file(options_path, options)



# embedder = SeqVecEmbedder()
embedder = ElmoEmbedder(options_file='./seqVec/options.json', weight_file='./seqVec/weights.hdf5', cuda_device=0)


total_num_seqs = 0
def lstm_features(seq):
    global total_num_seqs
    total_num_seqs += 1
    print("\rExtract feature for No.{} seq:".format(total_num_seqs),end=" ")
    print(seq)
    
    embedding = embedder.embed_sentence(seq)
    protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L x 1024]
    np_arr = protein_embd.cpu().detach().numpy()
    
    return np_arr

seq_all = []
#for dt_name in ['kiba','davis']:
for dt_name in ['pdb']:
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
    
    
np.save('./data/processed/seq_lstm-pdb.npy', seq_lstm)