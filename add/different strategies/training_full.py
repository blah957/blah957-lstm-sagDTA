#python training.py
import numpy as np
import pandas as pd
import os
import sys

from random import shuffle
#from sqlalchemy import collate
import torch
import torch.nn as nn

# from gat_gcn_sag_g import GAT_GCN_sagg
# from gat_gcn_sag_h import GAT_GCN_sagh
# from gcn3_sag_g import GCN3_GLOBAL
# from gcn3_sag_h import GCN3_HIER
from gat_gcn_h import GAT_GCN_h
from gat_gcn_g import GAT_GCN_g
from gat_gcn_h_max import GAT_GCN_h_max
from gat_gcn_h_mean import GAT_GCN_h_mean
from gat_gcn_g_max import GAT_GCN_g_max
from gat_gcn_g_mean import GAT_GCN_g_mean

from utils import *
from pt_data import create_data

os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"


datasets = [['davis','kiba'][int(sys.argv[1])]] 

# modeling = [GAT_GCN_sagg,GAT_GCN_sagh,GCN3_GLOBAL,GCN3_HIER,GAT_GCN_h,GAT_GCN_g][int(sys.argv[2])]  
modeling = [GAT_GCN_h,GAT_GCN_g,GAT_GCN_h_max,GAT_GCN_h_mean,GAT_GCN_g_max,GAT_GCN_g_mean][int(sys.argv[2])]  
model_st = modeling.__name__ 

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

if not os.path.isdir("./result"):
    os.mkdir("./result")
if not os.path.isdir("./checkpoint"):
    os.mkdir("./checkpoint")

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005 
LOG_INTERVAL = 10 
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = modeling().to(device)
# device_name = torch.cuda.get_device_name(device.index) if device.type == 'cuda' else 'CPU'
# print(('  using ' + device_name + ' ').center(80, '*')) 


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    train_data,test_data= create_data(dataset)
    
    # make data PyTorch mini-batch processing ready
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,collate_fn =collate)# DataLoader
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,collate_fn =collate)

    # training the model
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = 'train_model_' + model_st + '_' + dataset +  '.model'
    result_file_name = 'train_result_' + model_st + '_' + dataset +  '.csv'
    
    if not os.path.isdir("./result/train"):
        os.mkdir("./result/train")
    with open('./result/train/'+f'{result_file_name}','a+') as f:
        f.write('epoch,rmse,mse,pearson,spearman,ci\n')
        
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch+1)
        
        model.to(device)
        G,P = predicting(model, device, test_loader)
        
        ret = [epoch+1,rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
        with open('./result/train/'+f'{result_file_name}','a+') as f:
            f.write(','.join(map(str,ret)))
            f.write("\n")
                       
        if ret[2]<best_mse:
            torch.save(model.state_dict(), './result/train/'+f'{model_file_name}')
            best_mse = ret[2]
            best_ci = ret[-1]
            best_epoch = epoch+1
        
            
            # with open(result_file_name,'w') as f:
            #     f.write(','.join(map(str,ret)))
            print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
        else:
            print(ret[2],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
            
    

