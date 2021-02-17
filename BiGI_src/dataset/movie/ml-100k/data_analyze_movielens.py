import codecs
import json


def file_pair(in_file,out_file,in_file_after,out_file_after):
    trainset = []
    testset = []
    user_id = {}
    item_id = {}

    with codecs.open(in_file,"r",encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            if line[0] not in user_id:
                user_id[line[0]]=len(user_id)
            if line[1] not in item_id:
                item_id[line[1]]=len(item_id)
            trainset.append([line[0], line[1], line[2]])

    # print(len(user),len(item),len(trainset))
    cnt=0
    with codecs.open(out_file, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            # user.add(line[0])
            # item.add(line[1])
            if line[1] not in item_id:
                cnt+=1
                continue
            testset.append([line[0], line[1], line[2]])

    # print(len(user), len(item), len(testset)) # some cold item in test set
    print(in_file)
    print("user_len: ", len(user_id))
    print("item_len: ", len(item_id))
    print("train_set_len",len(trainset))
    print("test_set_len", len(testset))
    print("")

    with codecs.open(in_file_after, "w", encoding="utf-8") as fw:
        for mytuple in trainset:
            fw.write("{}\t{}\t{}\n".format(user_id[mytuple[0]],item_id[mytuple[1]],int(mytuple[2])))

    with codecs.open(out_file_after, "w", encoding="utf-8") as fw:
        for mytuple in testset:
            fw.write("{}\t{}\t{}\n".format(user_id[mytuple[0]],item_id[mytuple[1]],int(mytuple[2])))

if __name__ == '__main__':
    file_pair("u1.base","u1.test","1/rating_train_after.dat","1/rating_test_after.dat")
    file_pair("u2.base", "u2.test","2/rating_train_after.dat","2/rating_test_after.dat")
    file_pair("u3.base", "u3.test","3/rating_train_after.dat","3/rating_test_after.dat")
    file_pair("u4.base", "u4.test","4/rating_train_after.dat","4/rating_test_after.dat")
    file_pair("u5.base", "u5.test","5/rating_train_after.dat","5/rating_test_after.dat")


"""
u1.base
user_len:  943
item_len:  1650
train_set_len 80000
test_set_len 19968

u2.base
user_len:  943
item_len:  1648
train_set_len 80000
test_set_len 19964

u3.base
user_len:  943
item_len:  1650
train_set_len 80000
test_set_len 19964

u4.base
user_len:  943
item_len:  1660
train_set_len 80000
test_set_len 19973

u5.base
user_len:  943
item_len:  1650
train_set_len 80000
"""