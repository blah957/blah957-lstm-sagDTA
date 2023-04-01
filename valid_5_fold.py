#python train_start.py
import numpy as np
import pandas as pd
import os
import sys

from random import shuffle
#from sqlalchemy import collate
import torch
import torch.nn as nn

from gat_gcn_sag_g import GAT_GCN_sagg
from gat_gcn_sag_h import GAT_GCN_sagh
from gcn3_sag_g import GCN3_GLOBAL
from gcn3_sag_h import GCN3_HIER
from utils import *
from pt_data import create_data_5_fold

os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"


datasets = [['davis','kiba'][int(sys.argv[1])]] 

modeling = [GAT_GCN_sagg,GAT_GCN_sagh,GCN3_GLOBAL,GCN3_HIER][int(sys.argv[2])]  

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

fold = [ 1, 2, 3, 4, 5][int(sys.argv[4])]
cross_validation_flag = True
model_st = modeling.__name__



if not os.path.isdir("./result"):
    os.mkdir("./result")
if not os.path.isdir("./checkpoint"):
    os.mkdir("./checkpoint")

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005 #0.0005
LOG_INTERVAL = 10 #20？
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = modeling().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    train_data,test_data= create_data_5_fold(dataset,fold)
    print(fold)
    
    # make data PyTorch mini-batch processing ready
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,collate_fn =collate)# DataLoader
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,collate_fn =collate)

    # training the model
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    #model_file_name = 'train_model_' + model_st + '_' + dataset +  '.tar' #.tar替换了.model
    # model_file_name = 'train_model_' + model_st + '_' + dataset
    model_file_name = 'train_model_' + model_st + '_' + dataset+ '_' + 'fold-' + str(fold)
    result_file_name = 'train_result_' + model_st + '_' + dataset+ '_' + 'fold-' + str(fold) +  '.csv'
       
    if not os.path.isdir("./result/train"):
                os.mkdir("./result/train")
    with open('./result/train/'+f'{result_file_name}','a+') as f:
        f.write('epoch,rmse,mse,pearson,spearman,ci\n')
        
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch+1)
        G,P = predicting(model, device, test_loader)
        
        ret = [epoch+1,rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
        with open('./result/train/'+f'{result_file_name}','a+') as f:
            f.write(','.join(map(str,ret)))
            f.write("\n")
                            
        if ret[2]<best_mse:
            #torch.save(model.state_dict(), model_file_name)
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn
            }
            if not os.path.isdir('./checkpoint/train'):
                os.mkdir('./checkpoint/train')
            #torch.save(checkpoint, './checkpoint/'+f'{model_file_name}'+'/%s.pth' %(str(epoch+1)))
            torch.save(checkpoint, './checkpoint/train/'+f'{model_file_name}'+'_best.pth')

            best_mse = ret[2]
            best_ci = ret[-1]
            best_epoch = epoch+1
        
            print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
        else:
            print(ret[2],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
            