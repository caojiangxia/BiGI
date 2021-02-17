import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GCN import GCN
from torch.autograd import Variable

class GNN(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(GNN, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number):
            self.encoder.append(DGCNLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]
        self.score_function1 = nn.Linear(opt["hidden_dim"]+opt["hidden_dim"],10)
        self.score_function2 = nn.Linear(10,1)

    def forward(self, ufea, vfea, UV_adj, VU_adj,adj):
        learn_user = ufea
        learn_item = vfea
        for layer in self.encoder:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_item = F.dropout(learn_item, self.dropout, training=self.training)
            learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
        return learn_user, learn_item

class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        Item_ho = self.gc2(vfea, UV_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        Item = torch.cat((Item_ho, vfea), dim=1)
        User = self.user_union(User)
        Item = self.item_union(Item)
        return F.relu(User), F.relu(Item)