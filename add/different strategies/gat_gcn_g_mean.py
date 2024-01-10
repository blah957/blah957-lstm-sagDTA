import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv,SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
#from layers import SAGPool



class GAT_GCN_g_mean(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78,
                 n_filters=32, output_dim=128, dropout=0.1):#0.2-0.5

        super(GAT_GCN_g_mean, self).__init__()
        #self.pooling_ratio = 1.0

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.bn1 = nn.BatchNorm1d(num_features_xd*10)
    
        
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.bn2 = nn.BatchNorm1d(num_features_xd*10)
        
        #self.pool = SAGPooling(2 * num_features_xd*10, ratio=self.pooling_ratio, GNN=GCNConv)
        
        #self.fc_g1 = torch.nn.Linear(num_features_xd*10*2*2, 1500) #1560x1500
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500) 
        
        self.bn3 = nn.BatchNorm1d(1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.bn4 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.conv_xt_1 = nn.Conv1d(in_channels=640, out_channels=n_filters, kernel_size=8) 
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
        x1 = x
               
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x2 = x
        
        x = torch.cat([x1, x2], dim=1) #还是说去掉这个跟GraphDTA一样？
        
        #x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        
        #x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #x = gmp(x, batch) 
        x = gap(x, batch) 
        
        
        
        x = self.fc_g1(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        
        
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
