import codecs
import numpy as np
relation_class=set()
testset=[]
trainset=[]
user_id={}
item_id={}
feature30=[]
feature128=[]
def tofloat(t):
    ret=[]
    for i in t:
        ret.append(float(i))
    return ret
with codecs.open("rating_train.dat","r",encoding="utf-8") as fr:
    for line in fr:
        line = line.strip().split("\t")
        relation_class.add(int(line[2]))
        if line[0] not in user_id:
            user_id[line[0]] = len(user_id)
        if line[1] not in item_id:
            item_id[line[1]] = len(item_id)
        trainset.append(line)

print(len(user_id),len(item_id))

trainset_item_id_len=len(item_id)

with codecs.open("rating_test.dat","r",encoding="utf-8") as fr:
    for line in fr:
        line = line.strip().split("\t")
        if line[1] not in item_id:
            item_id[line[1]]=trainset_item_id_len
        testset.append(line)

print(len(user_id),len(item_id),trainset_item_id_len) # in test part, here have some cold-start item

for i in range(len(user_id)+trainset_item_id_len):
    feature30.append([])
    feature128.append([])
cnt=cntt=0
with codecs.open("vectors_u30.dat","r",encoding="utf-8") as fr:
    for line in fr:
        cntt+=1
        line=line.strip().split(" ")
        f = tofloat(line[1:])
        feature30[user_id[line[0]]] = f
with codecs.open("vectors_u.dat","r",encoding="utf-8") as fr:
    for line in fr:
        cnt+=1
        line=line.strip().split(" ")
        f = tofloat(line[1:])
        feature128[user_id[line[0]]] = f

with codecs.open("vectors_v30.dat","r",encoding="utf-8") as fr:
    for line in fr:
        cntt+=1
        line=line.strip().split(" ")
        f = tofloat(line[1:])
        feature30[item_id[line[0]]+len(user_id)] = f
with codecs.open("vectors_v.dat","r",encoding="utf-8") as fr:
    for line in fr:
        cnt+=1
        line=line.strip().split(" ")
        f = tofloat(line[1:])
        feature128[item_id[line[0]]+len(user_id)] = f
print(cnt,cntt)
feature30 = np.array(feature30)
feature128 = np.array(feature128)

print(feature30.shape,type(feature30[0][0]))
print(feature128.shape)

print("generate: rating_train_after_after.dat rating_test_after_after.dat feature_30.dat feature_128.dat! are you sure?")
input()
with codecs.open("rating_train_after.dat","w",encoding="utf-8") as fw:
    for mytuple in trainset:
        fw.write("{}\t{}\t{}\n".format(user_id[mytuple[0]],item_id[mytuple[1]],int(mytuple[2])))


with codecs.open("rating_test_after_after.dat","w",encoding="utf-8") as fw:
    for mytuple in testset:
        fw.write("{}\t{}\t{}\n".format(user_id[mytuple[0]],item_id[mytuple[1]],int(mytuple[2])))

np.savetxt("feature30.txt",feature30)
np.savetxt("feature128.txt",feature128)

'''
np.savetxt("mytensor.txt",out)
with open("mytensor.txt","r") as fr:
    A = np.loadtxt(fr)
'''