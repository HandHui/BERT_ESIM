import os
import pandas as pd
from collections import Counter

### 研究涉及的数据长度以及标签分布

### 研究sample中的内容数
path = 'data/round1/'

# smaple = 'sample_submission.csv'
# cons = pd.read_csv(path+smaple)
# print(type(cons))
# print("sample长度：",len(cons))  ##29822

####  全为0， 则F为0
#### 全为1，方案如下：
# cons.label = 1
# cons.to_csv('result.csv',index=False)
path1 = 'slB/'
traindata = 'train.txt'
validdata = 'valid.txt'
testdata = 'test_with_id.txt'

with open(path+path1+traindata,'r',encoding='utf-8') as tf:
    tlines = tf.readlines()
with open(path+path1+validdata,'r',encoding='utf-8') as vf:
    vlines = vf.readlines()
print("训练集短短相似度A集长度 : ",len(tlines))  ##9867
print("验证集短短相似度A集长度 : ",len(vlines))  ##1645
# for line in tlines:
#     print(line.strip())
#     break

with open(path+path1+testdata,'r',encoding='utf-8') as tf:
    telines = tf.readlines()
print("测试集短短相似度A集长度 : ",len(telines))   ##4934
# for line in telines:
#     line = line.strip()
#     print(line)
#     break

minl,maxl = 200,0
sum_len = 0
cnt = 0
con_sumlen = []
leg_more_100 = 0
label_true_maxlen = 0
sr_len = 0
i = 0
for line in tlines:
    line = line.strip()
    line = eval(line)
    source = line['source']
    target = line['target']
    sr_len = max(sr_len,min(len(source),len(target)))
    if abs(len(source)-len(target))>100:
        leg_more_100 += 1
    con_sumlen.append(len(source)+len(target))
    label = line['labelB']
    if label == '1':
        if len(source)>10 and len(source)+len(target) > 5000:
            i += 1
            if i>8:
                print(line)
                break
    # if label == '1':
    #     label_true_maxlen = max(label_true_maxlen,abs(len(source)-len(target)))
    #     if label_true_maxlen > 6000 and label_true_maxlen < 10000:
    #         print(line)
    #         break
    cnt += int(label)
    sum_len = max(sum_len,len(source)+len(target))
    # print(len(source)+len(target))
    minl = min(len(source),len(target),minl)
    maxl = max(len(source),len(target),maxl) 
    # if len(source)+len(target) > 700 and abs(len(source)-len(target))>100:
    #     print(line)
    #     break
    # if maxl>300 and abs(len(source)-len(target)) <50:
    #     print(line)
    #     break
    # if minl <= 3:
    #     print(line)
        # break
print('长度范围: ',minl,maxl)   ###[3,150]
print('总长度：',sum_len)  ##300
print(cnt)

span_len = [0]*8
for k,v in Counter(con_sumlen).items():
    idx = min(7,k//100)
    span_len[idx] += v

print('总长度区间(100为一区间): ')
print(span_len)
print('句子之间长度差超过100的个数：', leg_more_100)
print('相似句子之间的长度最大差：',label_true_maxlen)
print('source长度最大值: ',sr_len)