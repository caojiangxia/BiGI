import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class Attention(nn.Module):
    def __init__(self,opt):
        super(Attention, self).__init__()
        self.lin1 = nn.Linear(opt["hidden_dim"],opt["hidden_dim"])
        self.lin2 = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        self.opt = opt

    def forward(self, seq, key ,choose):
        if choose:
            seq = self.lin1(seq) # b * k * h
            key = self.lin2(key).unsqueeze(1) # b * 1 * h
        else :
            seq = self.lin2(seq)
            key = self.lin1(key).unsqueeze(1)

        value = torch.matmul(key, seq.transpose(-1,-2)) # b * 1 * k
        value /= math.sqrt(self.opt["hidden_dim"])
        value = F.softmax(value, dim=-1)
        answer = torch.matmul(value,seq) # b * 1 * h
        answer = answer.squeeze(1)

        return answer


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

class AttDGI(nn.Module):
    def __init__(self, opt):
        super(AttDGI, self).__init__()
        self.opt = opt
        self.read = AvgReadout()
        self.att = Attention(opt)
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
    def forward(self, user_hidden_out, item_hidden_out, real_user, real_item, fake_user, fake_item, real_item_id, real_user_id, fake_item_id, fake_user_id, msk=None, samp_bias1=None,
                samp_bias2=None):

        S_u_One = self.read(user_hidden_out, msk).unsqueeze(0)  # 1*hidden_dim
        S_i_One = self.read(item_hidden_out, msk).unsqueeze(0)  # 1*hidden_dim
        S_Two = self.lin(torch.cat((S_u_One, S_i_One), dim = -1)) # 1 * hidden_dim
        S_Two = self.sigm(S_Two)  # hidden_dim  need modify

        real_sub_u_Two = self.att(real_item_id, real_user, 0) + real_user  # hidden_dim
        real_sub_i_Two = self.att(real_user_id, real_item, 1) + real_item  # hidden_dim
        fake_sub_u_Two = self.att(fake_item_id, fake_user, 0) + fake_user  # hidden_dim
        fake_sub_i_Two = self.att(fake_user_id, fake_item, 1) + fake_item  # hidden_dim

        real_sub_Two = self.lin_sub(torch.cat((real_sub_u_Two, real_sub_i_Two),dim = 1))
        real_sub_Two = self.sigm(real_sub_Two)

        fake_sub_Two = self.lin_sub(torch.cat((fake_sub_u_Two, fake_sub_i_Two),dim = 1))
        fake_sub_Two = self.sigm(fake_sub_Two)

        real_sub_prob = self.disc(S_Two, real_sub_Two)
        fake_sub_prob = self.disc(S_Two, fake_sub_Two)


        prob = torch.cat((real_sub_prob, fake_sub_prob))
        label = torch.cat((torch.ones_like(real_sub_prob), torch.zeros_like(fake_sub_prob)))

        return prob, label
