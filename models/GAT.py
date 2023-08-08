from platform import node
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, AttentionalAggregation
import torch_geometric.utils as utils
import copy


class GAT(nn.Module):

    def __init__(self, config, num_class, x_dim, device):
        super(GAT, self).__init__()

        self.dropout_rate = config['dropout_p']
        self.hidden_size = config['hidden_size']
        self.n_layers = config["n_layers"]
        self.device = device
        alpha = 0.3
        self.embedding = nn.Embedding(30522, x_dim)
        
        self.context_LSTM = nn.LSTM(input_size=x_dim,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.n_layers,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=self.dropout_rate)
        
        self.gru = nn.GRU(input_size = x_dim,
                          hidden_size = self.hidden_size,
                          num_layers = self.n_layers, 
                          batch_first = True, 
                          bidirectional = True)
    

        self.gnn1 = GATv2Conv(x_dim, out_channels=self.hidden_size//8, dropout=self.dropout_rate, heads=8, negative_slope=alpha)
        self.gnn2 = GATv2Conv(self.hidden_size//8, self.hidden_size, dropout=self.dropout_rate, concat=False, negative_slope=alpha)
           
        
        self.BN = nn.BatchNorm1d(self.hidden_size)

        self.relu = nn.ReLU()
        self.pool = global_add_pool

        gate_nn=nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),
                                  nn.Tanh(),
                                  nn.Linear(self.hidden_size,1,bias=False))

        self.gate_nn = AttentionalAggregation(gate_nn)
        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_size, 100),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(100, 1 if num_class==2 else num_class)
        )

        self.reset_parameters()
    
    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
    
    def reset_parameters(self):
        for name, param in self.fc_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)

    def node_encoder(self, x, sentence_tokens=None):
        if sentence_tokens!=None:           
            X_feature = torch.tensor([]).to(self.device)
            for x0 in sentence_tokens:
                for x1 in x0:
                    x_emb = self.embedding(torch.tensor(x1).to(self.device))
                    x_encode = self.context_LSTM(x_emb)
                    X_feature = torch.cat((X_feature, torch.mean(x_encode[0], dim=0).unsqueeze(0)), 0)

        else:
            X_feature = x
        return X_feature

    def global_graph_encoding(self, x, edge_index, edge_attr=None, edge_atten=None):
        node_rep1 = self.gnn1(x=x, edge_index=edge_index, edge_attr=edge_atten)
        node_rep1 = self.relu(self.BN(self.dropout(node_rep1)))
        graph_output = self.gnn2(node_rep1, edge_index, edge_attr=edge_atten)
        graph_output  = self.relu(self.BN(graph_output))

        return graph_output

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None, sentence_tokens=None):

        x = self.node_encoder(x, sentence_tokens)

        X_global = self.global_graph_encoding(x, edge_index, edge_attr, edge_atten)
        X_feat = F.dropout(X_global, p = self.dropout_rate, training=self.training)
        out = self.gate_nn(X_feat, batch)
        output = self.fc_out(out)
        return output
    
    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None, sentence_tokens=None):
        x = self.node_encoder(x, sentence_tokens)
        X_feature = copy.deepcopy(x.detach())

        X_global = self.global_graph_encoding(x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
        X_feat = F.dropout(X_global, p = self.dropout_rate, training=self.training)

        return X_feat, X_feature

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.gate_nn(emb, batch))
