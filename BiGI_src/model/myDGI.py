import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GAT import GAT

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)


class Discriminator(nn.Module):
    def __init__(self, n_in,n_out):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_in, n_out, 1)
        self.sigm = nn.Sigmoid()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, S, node, s_bias=None):
        S = S.expand_as(node) # batch * hidden_dim
        score = torch.squeeze(self.f_k(node, S),1) # batch
        if s_bias is not None:
            score += s_bias

        return self.sigm(score)

class myDGI(nn.Module):
    def __init__(self, opt):
        super(myDGI, self).__init__()
        self.opt = opt
        self.read = AvgReadout()
        self.att = GAT(opt)
        self.sigm = nn.Sigmoid()
        self.lin = nn.Linear(opt["hidden_dim"] * 2, opt["hidden_dim"])
        self.lin_sub = nn.Linear(opt["hidden_dim"] * 2, opt["hidden_dim"])
        self.disc = Discriminator(opt["hidden_dim"],opt["hidden_dim"])
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    #
    def forward(self, user_hidden_out, item_hidden_out, fake_user_hidden_out, fake_item_hidden_out, UV_adj, VU_adj, CUV_adj, CVU_adj, user_One, item_One, msk=None, samp_bias1=None,
                samp_bias2=None):

        S_u_One = self.read(user_hidden_out, msk)  # hidden_dim
        S_i_One = self.read(item_hidden_out, msk)  # hidden_dim
        S_Two = self.lin(torch.cat((S_u_One, S_i_One)).unsqueeze(0)) # 1 * hidden_dim
        S_Two = self.sigm(S_Two)  # hidden_dim  need modify

        real_user, real_item = self.att(user_hidden_out, item_hidden_out, UV_adj, VU_adj)
        fake_user, fake_item = self.att(fake_user_hidden_out, fake_item_hidden_out, CUV_adj, CVU_adj)

        real_user_index_feature_Two = torch.index_select(real_user, 0, user_One)
        real_item_index_feature_Two = torch.index_select(real_item, 0, item_One)
        fake_user_index_feature_Two = torch.index_select(fake_user, 0, user_One)
        fake_item_index_feature_Two = torch.index_select(fake_item, 0, item_One)
        real_sub_Two = self.lin_sub(torch.cat((real_user_index_feature_Two, real_item_index_feature_Two),dim = 1))
        real_sub_Two = self.sigm(real_sub_Two)

        fake_sub_Two = self.lin_sub(torch.cat((fake_user_index_feature_Two, fake_item_index_feature_Two),dim = 1))
        fake_sub_Two = self.sigm(fake_sub_Two)

        real_sub_prob = self.disc(S_Two, real_sub_Two)
        fake_sub_prob = self.disc(S_Two, fake_sub_Two)

        prob = torch.cat((real_sub_prob, fake_sub_prob))
        label = torch.cat((torch.ones_like(real_sub_prob), torch.zeros_like(fake_sub_prob)))

        return prob, label
