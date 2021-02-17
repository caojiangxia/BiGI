import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import copy

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def Bernoulli(rate):
    num = random.randint(0,1000000)
    p = num/1000000
    if p < rate:
        return 1
    else :
        return 0

def struct_corruption(opt, original_dict, rate):
    adj_dict = copy.deepcopy(original_dict)
    UV_edges = []
    VU_edges = []
    all_edges = []
    user_fake_dict = {}
    item_fake_dict = {}
    corruption_edges = int(opt["number_user"] * opt["number_item"] * rate)+1
    print("corruption_edges: ", corruption_edges)

    for k in range(corruption_edges):
        i = random.randint(0, opt["number_user"]-1)
        j = random.randint(0, opt["number_item"]-1)

        if adj_dict[i].get(j, "zxczxc") is "zxczxc":  # if 1 : adj is 0
            adj_dict[i][j] = 1
        else :
            del adj_dict[i][j]


    for i in adj_dict.keys():
        user_fake_dict[i] = set()
        for j in adj_dict[i].keys():
            UV_edges.append([i,j])
            all_edges.append([i,j + opt["number_user"]])
            all_edges.append([j + opt["number_user"] , i])
            user_fake_dict[i].add(j)
            VU_edges.append([j, i])
            if j not in item_fake_dict.keys():
                item_fake_dict[j] = set()
            item_fake_dict[j].add(i)

    UV_edges = np.array(UV_edges)
    VU_edges = np.array(VU_edges)
    all_edges = np.array(all_edges)
    UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                           shape=(opt["number_user"], opt["number_item"]),
                           dtype=np.float32)
    VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                           shape=(opt["number_item"], opt["number_user"]),
                           dtype=np.float32)

    all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),
                            shape=(opt["number_item"] + opt["number_user"], opt["number_item"] + opt["number_user"]),
                            dtype=np.float32)

    UV_adj = normalize(UV_adj)
    VU_adj = normalize(VU_adj)
    all_adj = normalize(all_adj)
    UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
    VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
    all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)
    return UV_adj,VU_adj,all_adj,user_fake_dict,item_fake_dict

class GraphMaker(object):
    def __init__(self, opt, filename):
        self.opt = opt
        self.user = set()
        self.item = set()
        data=[]
        with codecs.open(opt["data_dir"] + filename) as infile:
            for line in infile:
                line = line.strip().split("\t")
                data.append((int(line[0]),int(line[1]),float(line[2])))
                self.user.add(int(line[0]))
                self.item.add(int(line[1]))

        opt["number_user"] = len(self.user)
        opt["number_item"] = len(self.item)
        self.raw_data = data
        self.UV,self.VU, self.adj, self.corruption_UV,self.corruption_VU, self.fake_adj = self.preprocess(data,opt)

    def preprocess(self,data,opt):
        UV_edges = []
        VU_edges = []
        all_edges = []
        real_adj = {}

        user_real_dict = {}
        item_real_dict = {}
        for edge in data:
            UV_edges.append([edge[0],edge[1]])
            if edge[0] not in user_real_dict.keys():
                user_real_dict[edge[0]] = set()
            user_real_dict[edge[0]].add(edge[1])

            VU_edges.append([edge[1], edge[0]])
            if edge[1] not in item_real_dict.keys():
                item_real_dict[edge[1]] = set()
            item_real_dict[edge[1]].add(edge[0])

            all_edges.append([edge[0],edge[1] + opt["number_user"]])
            all_edges.append([edge[1] + opt["number_user"], edge[0]])
            if edge[0] not in real_adj :
                real_adj[edge[0]] = {}
            real_adj[edge[0]][edge[1]] = 1

        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(opt["number_user"], opt["number_item"]),
                               dtype=np.float32)
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(opt["number_item"], opt["number_user"]),
                               dtype=np.float32)
        all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),shape=(opt["number_item"]+opt["number_user"], opt["number_item"]+opt["number_user"]),dtype=np.float32)
        UV_adj = normalize(UV_adj)
        VU_adj = normalize(VU_adj)
        all_adj = normalize(all_adj)
        UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)

        print("real graph loaded!")
        corruption_UV_adj, corruption_VU_adj, fake_adj, user_fake_dict,item_fake_dict = struct_corruption(opt,real_adj,opt["struct_rate"])

        self.user_real_dict = user_real_dict
        self.user_fake_dict = user_fake_dict

        self.item_real_dict = item_real_dict
        self.item_fake_dict = item_fake_dict
        print("fake graph loaded!")
        return UV_adj,VU_adj, all_adj, corruption_UV_adj, corruption_VU_adj,fake_adj

