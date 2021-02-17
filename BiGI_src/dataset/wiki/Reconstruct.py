import codecs
import json
import random
def all_train_set(wiki_train,wiki_test):
    dataset=[]
    with codecs.open(wiki_train,"r",encoding="utf-8") as fr :
        for line in fr:
            line = line.strip().split("\t")
            dataset.append(line)
    with codecs.open(wiki_test,"r",encoding="utf-8") as fr :
        for line in fr:
            line = line.strip().split("\t")
            dataset.append(line)
    return dataset

def divide_dataset(div,dataset,index):
    random.shuffle(index)
    train_len = int(len(index)*(float(div)/10))+1
    dataset = [dataset[i] for i in index]
    train_set = dataset[:train_len]
    test_set = dataset[train_len:]
    print(len(train_set),len(test_set))
    adj={}
    item_id= {}
    user_id = {}
    for mytuple in train_set:
        if mytuple[0] not in user_id:
            user_id[mytuple[0]]=len(user_id)
            adj[ user_id[mytuple[0]]]={}
        if mytuple[1] not in item_id:
            item_id[mytuple[1]]=len(item_id)
        adj[user_id[mytuple[0]]][item_id[mytuple[1]]] = 1
    for mytuple in test_set:
        if mytuple[0] not in user_id:
            continue
        if mytuple[1] not in item_id:
            continue
        adj[user_id[mytuple[0]]][item_id[mytuple[1]]] = 1

    print("you should make ", div, " folder manually!")
    print("user_number: ", len(user_id))
    print("item_number: ", len(item_id))
    with codecs.open(str(div)+"/rating_train_after.dat", "w", encoding="utf-8") as fw:
        for mytuple in train_set:
            fw.write("{}\t{}\t{}\n".format(user_id[mytuple[0]], item_id[mytuple[1]], int(mytuple[2])))

    with codecs.open(str(div)+"/case_test_after.dat", "w", encoding="utf-8") as fw:
        for mytuple in test_set:
            if mytuple[0] not in user_id:
                continue
            if mytuple[1] not in item_id:
                continue
            fw.write("{}\t{}\t{}\n".format(user_id[mytuple[0]], item_id[mytuple[1]], int(mytuple[2])))

            neg=random.randint(0,len(item_id)-1)
            while adj[user_id[mytuple[0]]].get(neg,"0") == 1:
                neg = random.randint(0, len(item_id))
            fw.write("{}\t{}\t{}\n".format(user_id[mytuple[0]], neg, 0))

if __name__ == '__main__':
    random.seed(10)
    dataset=all_train_set("rating_train.dat","rating_test.dat")
    index=[]
    for i,tuple in enumerate(dataset):
        index.append(i)
    divide_dataset(4,dataset,index)
    divide_dataset(4.5, dataset, index)
    divide_dataset(5, dataset, index)
    divide_dataset(5.5, dataset, index)


'''
25639 38456
you should make  4  folder manually!
user_number:  10361
item_number:  2001
28843 35252
you should make  4.5  folder manually!
user_number:  10965
item_number:  2148
32048 32047
you should make  5  folder manually!
user_number:  11549
item_number:  2258
35253 28842
you should make  5.5  folder manually!
user_number:  12008
item_number:  2365
'''


'''
25639 38456
you should make  4  folder manually!
user_number:  10361
item_number:  2001
28843 35252
you should make  4.5  folder manually!
user_number:  10984
item_number:  2149
32048 32047
you should make  5  folder manually!
user_number:  11488
item_number:  2259
35253 28842
you should make  5.5  folder manually!
user_number:  12082
item_number:  2352
'''