import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv,SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
#from layers import SAGPool

# GCN-CNN based model

class GAT_GCN_h(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78,
                 n_filters=32, output_dim=128, dropout=0.1):#0.2-0.5

        super(GAT_GCN_h, self).__init__()
        #self.pooling_ratio = 1.0

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.bn1 = nn.BatchNorm1d(num_features_xd*10)
        #self.pool1 = SAGPooling(num_features_xd*10, ratio=self.pooling_ratio, GNN=GCNConv)
        
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.bn2 = nn.BatchNorm1d(num_features_xd*10)
        #self.pool2 = SAGPooling(num_features_xd*10, ratio=self.pooling_ratio, GNN=GCNConv)
        
        # self.pool = SAGPooling(2 * num_features_xd*10, ratio=self.pooling_ratio, GNN=GCNConv)
        
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500) 
     
        self.bn3 = nn.BatchNorm1d(1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.bn4 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        #self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=640, out_channels=n_filters, kernel_size=8) #8改成了9
        self.bn_xt1 = nn.BatchNorm1d(n_filters)
        self.fc1_xt = nn.Linear(32*1017, output_dim)
        self.bn6 = nn.BatchNorm1d(output_dim)
        
        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        #x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        
        # apply global max pooling (gmp) and global mean pooling (gap)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) 
        
               
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        #x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)#1560 = 780*2(gmp+gap) 
       
        # x_g = torch.cat([x1_g, x2_g], dim=1)
        # print(x_g.shape) #torch.Size([16281, 1560])1560=78*10(注意头）*2（cat）
        # x_g, edge_index, _, batch, _, _ = self.pool(x_g, edge_index, None, batch)
        # x_g = torch.cat([gmp(x_g, batch), gap(x_g, batch)], dim=1)
        # print(x_g.shape) #[512, 3120] 3120 = 1560*2（gmp、gap cat)
        
        x = x1+x2 
        
        
        # print(x.shape) #[512, 1560]
        # x = torch.cat([x1+x2, x_g], dim=1) #从两个注意力里面选择最好的一个 #512x4680
        
        
        x = self.fc_g1(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        
        # # apply global max pooling (gmp) and global mean pooling (gap)
        # x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # x = self.relu(self.fc_g1(x))
        # x = self.dropout(x)
        # x = self.fc_g2(x)

        #embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(target)
        # flatten
        xt = conv_xt.view(-1, 32 * 1017)
        xt = self.fc1_xt(xt)
        xt = self.relu(xt)
        xt = self.bn6(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        
        
        # add some dense layers
        xc = self.fc1(xc)
        
        xc = self.relu(xc)
        xc = self.bn7(xc)
        xc = self.dropout(xc)
        
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.bn8(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

