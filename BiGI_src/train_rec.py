import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.trainer import DGITrainer
from utils.loader import DataLoader
from utils.GraphMaker import GraphMaker
from utils import torch_utils, helper
from utils.scorer import *
import json
import codecs
# torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--data_dir', type=str, default='dataset/dblp/')
parser.add_argument('--weight', action='store_true', default=False, help='Using weight graph?')

# model part
parser.add_argument('--sparse', action='store_true', default=False, help='GNN with sparse version or not.')
parser.add_argument('--GNN', type=int, default=1, help="The layer of encoder.")
parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
parser.add_argument('--dropout', type=float, default=0.3, help='GNN layer dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--lambda', type=float, default=0.3)
parser.add_argument('--negative', type=int, default=1)
parser.add_argument('--DGI', action='store_false', default=True)
parser.add_argument('--attention', action='store_false', default=True, help='Using attention in sub-graph?')
parser.add_argument('--struct', action='store_false', default=True, help='Using struct corruption in graph?')
parser.add_argument('--struct_rate', type=float, default=0.0001)
# train part
parser.add_argument('--num_epoch', type=int, default=200, help='Number of total training epochs.')
parser.add_argument('--min_neighbor', type=int, default=1, help='Number of max neighbor per node')
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--topn', type=int, default=10, help='Recommendation top-n item for user in test session')
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--save_node_feature', action='store_true', default=False, help='save node feature')
parser.add_argument('--wiki', type=bool, default=False, help='wiki-dataset')

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parser.parse_args()
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()
# make opt
opt = vars(args)
seed_everything(opt["seed"])


# load data adj-matrix; Now sparse tensor ,But not setting in gpu
G = GraphMaker(opt,"train.txt")
UV = G.UV
VU = G.VU
adj = G.adj
corruption_UV = G.corruption_UV
corruption_VU = G.corruption_VU
fake_adj = G.fake_adj
user_real_dict = G.user_real_dict
user_fake_dict = G.user_fake_dict
item_real_dict = G.item_real_dict
item_fake_dict = G.item_fake_dict
print("graph loaded!")

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)


# feature; Now numpy, if have side information.
user_feature = np.random.randn(opt["number_user"], opt["feature_dim"])
item_feature = np.random.uniform(-1, 1, (opt["number_item"], opt["feature_dim"]))
user_feature = torch.tensor(user_feature, dtype=torch.float32)
item_feature = torch.tensor(item_feature, dtype=torch.float32)


if opt["cuda"]:
    user_feature = user_feature.cuda()
    item_feature = item_feature.cuda()
    UV = UV.cuda()
    VU = VU.cuda()
    adj = adj.cuda()
    fake_adj = fake_adj.cuda()
    corruption_UV = corruption_UV.cuda()
    corruption_VU = corruption_VU.cuda()

print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + 'train.txt', opt['batch_size'], opt,
                         user_real_dict, user_fake_dict, item_real_dict, item_fake_dict, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + 'test.txt', opt["batch_size"], opt, user_real_dict, user_fake_dict, item_real_dict, item_fake_dict, evaluation=True)


# model
if not opt['load']:
    trainer = DGITrainer(opt)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = DGITrainer(opt)
    trainer.load(model_file)

dev_score_history = [0]
current_lr = opt['lr']
global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']


# start training
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    start_time = time.time()
    for i, batch in enumerate(train_batch):
        global_step += 1
        loss = trainer.reconstruct(UV, VU, adj, corruption_UV, corruption_VU, fake_adj, user_feature, item_feature, batch)  # [ [user_list], [item_list], [neg_item_list] ]
        train_loss += loss
    duration = time.time() - start_time
    print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                    opt['num_epoch'], train_loss/len(train_batch), duration, current_lr))
    print("batch_rec_loss: ", sum(trainer.epoch_rec_loss)/len(trainer.epoch_rec_loss))
    print("batch_dgi_loss: ", sum(trainer.epoch_dgi_loss) / len(trainer.epoch_dgi_loss))
    trainer.epoch_rec_loss = []
    trainer.epoch_dgi_loss = []
    if epoch % 5:
        continue
    # eval model
    print("Evaluating on dev set...")

    trainer.model.eval()
    trainer.update_bipartite(user_feature, item_feature, UV, VU, adj)

    precision_list1 = []
    recall_list1 = []
    ap_list1 = []
    ndcg_list1 = []
    rr_list1 = []

    precision_list3 = []
    recall_list3 = []
    ap_list3 = []
    ndcg_list3 = []
    rr_list3 = []

    precision_list5 = []
    recall_list5 = []
    ap_list5 = []
    ndcg_list5 = []
    rr_list5 = []

    precision_list10 = []
    recall_list10 = []
    ap_list10 = []
    ndcg_list10 = []
    rr_list10 = []

    for i, batch in enumerate(dev_batch):
        recommend_list_candidate = trainer.predict(batch)
        now_list = batch[0].numpy()
        for id,now in enumerate(now_list):
            recommend_list = []
            for i in recommend_list_candidate[id]:
                if i in train_batch.user_real_dict[now]:
                    continue
                else:
                    recommend_list.append(i)
                if len(recommend_list) == opt["topn"]:
                    break
            recommend_list = np.array(recommend_list)

            ALL_group_list = batch[1][id]
            pre, rec = precision_and_recall(recommend_list, ALL_group_list)
            ap = AP(recommend_list, ALL_group_list)
            rr = RR(recommend_list, ALL_group_list)
            ndcg = nDCG(recommend_list, ALL_group_list)

            add_metric(recommend_list[:1], ALL_group_list, precision_list1, recall_list1, ap_list1, rr_list1,
                       ndcg_list1)
            add_metric(recommend_list[:3], ALL_group_list, precision_list3, recall_list3, ap_list3, rr_list3,
                       ndcg_list3)
            add_metric(recommend_list[:5], ALL_group_list, precision_list5, recall_list5, ap_list5, rr_list5, ndcg_list5)
            add_metric(recommend_list[:10], ALL_group_list, precision_list10, recall_list10, ap_list10, rr_list10,
                       ndcg_list10)

    cal_metric(precision_list1, recall_list1, ap_list1, rr_list1, ndcg_list1)
    cal_metric(precision_list3, recall_list3, ap_list3, rr_list3, ndcg_list3)
    cal_metric(precision_list5, recall_list5, ap_list5, rr_list5, ndcg_list5)
    precision,recall,f1,mndcg,mmap,mmrr = cal_metric(precision_list10, recall_list10, ap_list10, rr_list10, ndcg_list10)

    train_loss = train_loss / train_batch.num_examples * opt['batch_size']  # avg loss per batch
    print(
        "epoch {}: train_loss = {:.6f}, dev_precision = {:.6f}, dev_recall = {:.6f}, dev_f1 = {:.4f}, NDCG = {:.6f}, MAP = {:.6f}, MRR = {:.6f}".format(
            epoch, \
            train_loss, precision, recall, f1, mndcg, mmap, mmrr))
    dev_score = f1
    file_logger.log(
        "{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_score, max([dev_score] + dev_score_history)))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    if epoch == 1 or dev_score > max(dev_score_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")

        best_metric = [precision_list1, recall_list1, ap_list1, rr_list1, ndcg_list1, precision_list3, recall_list3,
                       ap_list3, rr_list3, ndcg_list3, precision_list5, recall_list5, ap_list5, rr_list5, ndcg_list5,
                       precision_list10, recall_list10, ap_list10, rr_list10,
                       ndcg_list10]
        metric_name = opt["id"] + ".json"
        json.dump(best_metric, codecs.open(metric_name, "w", encoding="utf-8"))

        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}" \
                        .format(epoch, f1 * 100, mmap * 100, mmrr * 100, mndcg * 100))
        if opt["save_node_feature"]:
            user_hidden_out = trainer.user_hidden_out
            item_hidden_out = trainer.item_hidden_out
            bi_feature = torch.cat((user_hidden_out, item_hidden_out), dim=0).detach().cpu().numpy()
            np.savetxt("Bipartite_feature.txt" + str(opt["id"]), bi_feature)
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)

    # lr schedule
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]
    print("")

print("Training ended with {} epochs.".format(epoch))
if opt["save_node_feature"]:
    np.savetxt("Bipartite_feature.txt" + str(opt["id"]), bi_feature)

"""
CUDA_VISIBLE_DEVICES=1 nohup python -u train_rec.py --id dblp --struct_rate 0.00001 --GNN 2 > BiGIdblp.log 2>&1&

CUDA_VISIBLE_DEVICES=1 nohup python -u train_rec.py --data_dir dataset/movie/ml-100k/1/ --batch_size 128 --id ml100k --struct_rate 0.0001 --GNN 2 > BiGI100k.log 2>&1&

CUDA_VISIBLE_DEVICES=1 nohup python -u train_rec.py --batch_size 100000 --data_dir dataset/movie/ml-10m/ml-10M100K/1/ --id ml10m --struct_rate 0.00001 > BiGI10m.log 2>&1&
"""