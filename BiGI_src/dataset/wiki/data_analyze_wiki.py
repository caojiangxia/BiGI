import codecs
import numpy as np

relation_class = set()
testset = []
trainset = []
user_id = {}
item_id = {}

user_set=set()
item_set=set()
cnt = 0
with codecs.open("rating_train.dat", "r", encoding="utf-8") as fr:
    for line in fr:
        line = line.strip().split("\t")
        user_set.add(line[0])
        item_set.add(line[1])
        cnt+=1
with codecs.open("rating_test.dat", "r", encoding="utf-8") as fr:
    for line in fr:
        line = line.strip().split("\t")
        user_set.add(line[0])
        item_set.add(line[1])
        cnt+=1

print("user_num :",len(user_set))
print("item_num :",len(item_set))
print("edges :",cnt)

"""
def tofloat(t):
    ret = []
    for i in t:
        ret.append(float(i))
    return ret

with codecs.open("rating_train.dat", "r", encoding="utf-8") as fr:
    for line in fr:
        line = line.strip().split("\t")
        relation_class.add(int(line[2]))
        if line[0] not in user_id:
            user_id[line[0]] = len(user_id)
        if line[1] not in item_id:
            item_id[line[1]] = len(item_id)
        trainset.append(line)

print(len(user_id), len(item_id),len(relation_class))

trainset_item_id_len = len(item_id)

cnt=cntt=0
with codecs.open("case_test.dat","r",encoding="utf-8") as fr:
    for line in fr:
        line = line.strip().split("\t")
        if line[1] not in item_id:
            pass
            # item_id[line[1]]=trainset_item_id_len
            if int(line[2]) == 1:
                cnt+=1
            else :
                cntt+=1
        else :
            testset.append(line)

print(len(user_id), len(item_id), trainset_item_id_len)  # in test part, here have some cold-start item. all 685

print(
    "generate: rating_train_after.dat rating_test_after_after.dat! are you sure?")
input()
with codecs.open("rating_train_after.dat", "w", encoding="utf-8") as fw:
    for mytuple in trainset:
        fw.write("{}\t{}\t{}\n".format(user_id[mytuple[0]], item_id[mytuple[1]], int(mytuple[2])))

with codecs.open("rating_test_after_after.dat", "w", encoding="utf-8") as fw:
    for mytuple in testset:
        fw.write("{}\t{}\t{}\n".format(user_id[mytuple[0]], item_id[mytuple[1]], int(mytuple[2])))

"""

'''
np.savetxt("mytensor.txt",out)
with open("mytensor.txt","r") as fr:
    A = np.loadtxt(fr)
'''
