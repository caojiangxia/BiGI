import math

from sklearn import metrics
from sklearn.metrics import average_precision_score,auc,precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i+1
        dcg += 1/ math.log(rank+1, 2)
    return dcg / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i+2, 2)
    return idcg

def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i+1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0

def RR(ranked_list, ground_list):

    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0

def precision_and_recall(ranked_list,ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits/(1.0 * len(ranked_list))
    rec = hits/(1.0 * len(ground_list))
    return pre, rec

def ROCPR(y_test,y_pred_est):
    '''
    :param y_test: label
    :param y_pred_est: predict score
    :return:
    '''
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_est)
    average_precision = average_precision_score(y_test, y_pred_est)
    # print('link prediction metrics: AUC_ROC : %0.4f, AUC_PR : %0.4f' % (round(auc_roc, 4), round(auc_pr, 4)))
    return metrics.auc(fpr, tpr), average_precision



def add_metric(recommend_list,ALL_group_list,precision_list,recall_list,ap_list,rr_list,ndcg_list):
    pre, rec = precision_and_recall(recommend_list, ALL_group_list)
    ap = AP(recommend_list, ALL_group_list)
    rr = RR(recommend_list, ALL_group_list)
    ndcg = nDCG(recommend_list, ALL_group_list)
    precision_list.append(pre)
    recall_list.append(rec)
    ap_list.append(ap)
    rr_list.append(rr)
    ndcg_list.append(ndcg)

def cal_metric(precision_list,recall_list,ap_list,rr_list,ndcg_list):
    precison = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    # print(precison, recall)
    f1 = 2 * precison * recall / (precison + recall + 0.00000001)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    print("f:{} ndcg:{} map:{} mrr:{}".format(f1,mndcg,map,mrr))
    return precison,recall,f1,mndcg,map,mrr


def link_prediction_logistic(X_train,y_train,X_test,y_test):
    lg = LogisticRegression(penalty='l2', C=0.001,max_iter=500)
    lg.fit(X_train,y_train)
    lg_y_pred_est = lg.predict_proba(X_test)[:,1]
    fpr,tpr,thresholds = metrics.roc_curve(y_test,lg_y_pred_est)
    average_precision = average_precision_score(y_test, lg_y_pred_est)
    return metrics.auc(fpr,tpr), average_precision, lg_y_pred_est.tolist()

