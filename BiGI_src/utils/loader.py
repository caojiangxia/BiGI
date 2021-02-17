"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, user_real_dict, user_fake_dict, item_real_dict, item_fake_dict, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.ma = {}
        with open(filename) as infile:
            data=[]
            for line in infile:
                line=line.strip().split("\t")
                data.append([int(line[0]),int(line[1])])
                if int(line[0]) not in self.ma.keys():
                    self.ma[int(line[0])] = set()
                self.ma[int(line[0])].add(int(line[1]))
        self.raw_data = data
        self.user_real_dict = user_real_dict
        self.user_fake_dict = user_fake_dict

        self.item_real_dict = item_real_dict
        self.item_fake_dict = item_fake_dict

        if not evaluation:
            data = self.preprocess(data, opt) # [[user,item] ... ]
        else :
            data = self.preprocess_for_predict() # [ [user, [gound_truth]] ]
        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))
    def preprocess_for_predict(self):
        processed=[]
        for user in range(self.opt["number_user"]):
            ground_truth=[]
            if user not in self.ma.keys():
                continue
            for item in self.ma[user]:
                if item >= self.opt["number_item"]:
                    continue
                ground_truth.append(item)
            if len(ground_truth) == 0:
                continue
            ground_truth=sorted(ground_truth)
            processed.append([user,ground_truth])
        return processed
    def preprocess(self, data, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        self.user_item_pair = []
        for mytuple in data:
            processed.append((mytuple[0],mytuple[1]))
            if len(self.user_real_dict[mytuple[0]]) > self.opt["min_neighbor"] and len(self.user_fake_dict[mytuple[0]]) > self.opt[
                "min_neighbor"] and len(self.item_real_dict[mytuple[1]]) > self.opt["min_neighbor"] and len(
                self.item_fake_dict[mytuple[1]]) > self.opt["min_neighbor"]:
                self.user_item_pair.append((mytuple[0],mytuple[1]))
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval :
            batch = list(zip(*batch))
            return torch.LongTensor(batch[0]), batch[1]
        else :
            negative_tmp = []
            for i in range(batch_size):
                for j in range(self.opt["negative"]):
                    while 1:
                        rand = random.randint(0,self.opt["number_item"]-1)
                        if rand not in self.user_real_dict[batch[i][0]]:
                            negative_tmp.append((batch[i][0],rand))
                            break
            batch = list(zip(*batch))
            negative_tmp = list(zip(*negative_tmp))
            if self.opt["number_user"] * self.opt["number_item"] > 10000000:
                user_index = []
                item_index = []
                real_user_index_id = []
                fake_user_index_id = []
                real_item_index_id = []
                fake_item_index_id = []
                random.shuffle(self.user_item_pair)
                for id in range(10):
                    user = self.user_item_pair[id][0]
                    item = self.user_item_pair[id][1]
                    real_item_id = list(self.user_real_dict[user])
                    real_user_id = list(self.item_real_dict[item])

                    fake_item_id = list(self.user_fake_dict[user])
                    fake_user_id = list(self.item_fake_dict[item])
                    random.shuffle(real_item_id)
                    random.shuffle(fake_item_id)
                    random.shuffle(real_user_id)
                    random.shuffle(fake_user_id)
                    real_item_id = real_item_id[:self.opt["min_neighbor"]]
                    fake_item_id = fake_item_id[:self.opt["min_neighbor"]]
                    real_user_id = real_user_id[:self.opt["min_neighbor"]]
                    fake_user_id = fake_user_id[:self.opt["min_neighbor"]]
                    user_index.append(user)
                    item_index.append(item)

                    fake_user_id = real_user_id
                    fake_item_id = real_item_id

                    real_item_index_id.append(real_item_id)
                    real_user_index_id.append(real_user_id)
                    fake_item_index_id.append(fake_item_id)
                    fake_user_index_id.append(fake_user_id)
                return torch.LongTensor(batch[0]), torch.LongTensor(batch[1]) , torch.LongTensor(negative_tmp[1]) , torch.LongTensor(user_index), torch.LongTensor(item_index), torch.LongTensor(real_user_index_id), torch.LongTensor(fake_user_index_id), torch.LongTensor(real_item_index_id), torch.LongTensor(fake_item_index_id)
            return torch.LongTensor(batch[0]), torch.LongTensor(batch[1]),torch.LongTensor(negative_tmp[1])
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)



class wikiDataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, user_real_dict, user_fake_dict, item_real_dict, item_fake_dict, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.ma = {}
        with open(filename) as infile:
            data=[]
            for line in infile:
                line=line.strip().split("\t")
                data.append([int(line[0]),int(line[1]),int(line[2])])
                if int(line[0]) not in self.ma.keys():
                    self.ma[int(line[0])] = set()
                self.ma[int(line[0])].add(int(line[1]))
        self.raw_data = data
        self.user_real_dict = user_real_dict
        self.user_fake_dict = user_fake_dict

        self.item_real_dict = item_real_dict
        self.item_fake_dict = item_fake_dict

        data = self.preprocess(data, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)
        if not evaluation:
            data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        else :
            data = [data]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        self.user_item_pair = []
        for mytuple in data:
            processed.append((mytuple[0],mytuple[1],mytuple[2]))
            if len(self.user_real_dict[mytuple[0]]) > self.opt["min_neighbor"] and len(
                    self.user_fake_dict[mytuple[0]]) > self.opt[
                "min_neighbor"] and len(self.item_real_dict[mytuple[1]]) > self.opt["min_neighbor"] and len(
                self.item_fake_dict[mytuple[1]]) > self.opt["min_neighbor"]:
                self.user_item_pair.append((mytuple[0], mytuple[1]))
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval :
            batch = list(zip(*batch))
            return torch.LongTensor(batch[0]), torch.LongTensor(batch[1])+torch.tensor(self.opt["number_user"]), np.array(batch[2])
        else :
            negative_tmp = []
            for i in range(batch_size):
                for j in range(self.opt["negative"]):
                    while 1:
                        rand = random.randint(0,self.opt["number_item"]-1)
                        if rand not in self.user_real_dict[batch[i][0]]:
                            negative_tmp.append((batch[i][0],rand))
                            break
            batch = list(zip(*batch))
            negative_tmp = list(zip(*negative_tmp))
            if self.opt["number_user"] * self.opt["number_item"] > 10000000:
                user_index = []
                item_index = []
                real_user_index_id = []
                fake_user_index_id = []
                real_item_index_id = []
                fake_item_index_id = []
                random.shuffle(self.user_item_pair)
                for id in range(10):
                    user = self.user_item_pair[id][0]
                    item = self.user_item_pair[id][1]
                    real_item_id = list(self.user_real_dict[user])
                    real_user_id = list(self.item_real_dict[item])

                    fake_item_id = list(self.user_fake_dict[user])
                    fake_user_id = list(self.item_fake_dict[item])
                    random.shuffle(real_item_id)
                    random.shuffle(fake_item_id)
                    random.shuffle(real_user_id)
                    random.shuffle(fake_user_id)
                    real_item_id = real_item_id[:self.opt["min_neighbor"]]
                    fake_item_id = fake_item_id[:self.opt["min_neighbor"]]
                    real_user_id = real_user_id[:self.opt["min_neighbor"]]
                    fake_user_id = fake_user_id[:self.opt["min_neighbor"]]
                    user_index.append(user)
                    item_index.append(item)

                    fake_user_id = real_user_id
                    fake_item_id = real_item_id

                    real_item_index_id.append(real_item_id)
                    real_user_index_id.append(real_user_id)
                    fake_item_index_id.append(fake_item_id)
                    fake_user_index_id.append(fake_user_id)
                return torch.LongTensor(batch[0]), torch.LongTensor(batch[1]) , torch.LongTensor(negative_tmp[1]) , torch.LongTensor(user_index), torch.LongTensor(item_index), torch.LongTensor(real_user_index_id), torch.LongTensor(fake_user_index_id), torch.LongTensor(real_item_index_id), torch.LongTensor(fake_item_index_id)  # User , item, label -> batch | batch | batch
            return torch.LongTensor(batch[0]), torch.LongTensor(batch[1]),torch.LongTensor(negative_tmp[1]) # User , item,  neg_item -> batch | batch | batch
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

