import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GCN import GCN
from torch.autograd import Variable


class GNN2(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(GNN2, self).__init__()
        self.opt=opt

        self.layer_number = opt["GNN"]
        self.DGCN_1 = DGCNLayer(opt)
        self.DGCN_2 = DGCNLayer(opt)
        self.dropout = opt["dropout"]

        self.score_function1 = nn.Linear(opt["hidden_dim"]+opt["hidden_dim"],10)
        self.score_function2 = nn.Linear(10,1)

        self.user_index = torch.arange(0, self.opt["number_user"], 1)
        self.item_index = torch.arange(self.opt["number_user"], self.opt["number_user"] + self.opt["number_item"], 1)
        if self.opt["cuda"]:
            self.user_index = self.user_index.cuda()
            self.item_index = self.item_index.cuda()


    def score(self, fea):
        out = self.score_function1(fea)
        out = F.relu(out)
        out = self.score_function2(out)
        # out = F.relu(out)
        out = torch.sigmoid(out)
        return out.view(-1)

    def forward(self, user_fea, item_fea, UV, VU, adj):
        fea = torch.cat((user_fea,item_fea), dim = 0)

        fea = F.dropout(fea, self.dropout, training=self.training)
        out = self.DGCN_1(fea, adj)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.DGCN_2(out, adj)

        user = torch.index_select(out, 0, self.user_index)
        item = torch.index_select(out, 0, self.item_index)

        return user, item

class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """

    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt = opt
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
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.Union = nn.Linear(opt["hidden_dim"] + opt["feature_dim"], opt["hidden_dim"])

        self.user_index = torch.arange(0, self.opt["number_user"], 1)
        self.item_index = torch.arange(self.opt["number_user"], self.opt["number_user"] + self.opt["number_item"],
                                       1)
        if self.opt["cuda"]:
            self.user_index = self.user_index.cuda()
            self.item_index = self.item_index.cuda()
    def forward(self, fea, adj):
        user = self.gc1(fea, adj)
        item = self.gc2(fea, adj)
        # user = F.dropout(user, self.opt["dropout"], training=self.training)
        # item = F.dropout(item, self.opt["dropout"], training=self.training)
        after_user_item = torch.cat(
            (torch.index_select(user, 0, self.user_index), torch.index_select(item, 0, self.item_index)), dim=0)

        User_ho = self.gc3(after_user_item, adj)
        Item_ho = self.gc4(after_user_item, adj)

        after_user_item_ho = torch.cat(
            (torch.index_select(User_ho, 0, self.user_index), torch.index_select(Item_ho, 0, self.item_index)), dim=0)

        after_user_item_ho_X = torch.cat((after_user_item_ho, fea), dim=1)

        output = self.Union(after_user_item_ho_X)

        return F.relu(output)