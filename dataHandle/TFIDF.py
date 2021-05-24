import gensim
import json
import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle

from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora

### 根据tfidf值获取前n个关键字,sklearn.feature_extraction.text.TfidfVectorizer
### 数据加载在对应目录下，按需更改数据目录


# ### 模型及其存储
# variants = [
#             # 'ssA',
#             # 'ssB',
#             # 'slA',
#             # 'slB',
#             # 'llA',
#             'llB',
#         ]
# for var in variants:
#     fs = [
#             '../data/round1/%s/train.txt'%var,
#             # '../data/round2/%s.txt'%var,
#             # '../data/round3/divided_20210419/%s/train.txt'%var,
#             # '../data/rematch/%s/train.txt'%var,
#         ]
#     for f in fs: ##  = '../data/round1/%s/train.txt'%var
#         texts = []
#         with open(f,encoding='utf-8') as f:
#             lines = f.readlines()
#             for l in tqdm(lines[:1000]):
#                 l = l.strip()
#                 l = json.loads(l)
#                 texts.append(jieba.lcut(l['source']))
#                 texts.append(jieba.lcut(l['target']))
# document = [" ".join(sent0) for sent0 in texts]


# cv = TfidfVectorizer()
# tfidf_model = cv.fit(document)
# word2id = tfidf_model.vocabulary_

# output = open('tfmodel.pkl', 'wb')
# # input = open('model.pkl', 'rb')
# s = pickle.dump(tfidf_model, output)
# output.close()
# tfidf_model = pickle.load(input)
# input.close()
# # print clf2.predict(X[0:1])


# sparse_result = tfidf_model.transform(document[:3])
# sort = np.argsort(sparse_result.toarray(),axis=1)[:,-20:]
# names = tfidf_model.get_feature_names()
# kw = pd.Index(names)[sort]
# print(kw)
# print(document[0])
# dictionary = corpora.Dictionary(texts)

# word2id = dictionary.token2id
# id2word = {v:k for k,v in word2id.items()}
# corpus = [dictionary.doc2bow(text) for text in texts]

# tfidf = gensim.models.tfidfmodel.TfidfModel(corpus)

# print(len(texts) == len(corpus_tfidf))
# source = jieba.lcut(texts[0])
# target = jieba.lcut(texts[1])
# cs = [dictionary.doc2bow(text) for text in [source,target]]
# cs_tfidf = tfidf[cs]
# d = {}
# for doc in cs_tfidf:
#     for id, value in doc:
#         word = dictionary.get(id)
#         d[word] = value

def new_sent(d0,k0):
    con_lst = []
    if len(d0)<510:
        return d0
    for c in jieba.lcut(d0):
        if c in k0:
            con_lst.append(c)
    return ''.join(con_lst)
    # print(''.join(con_lst))

input = open('tfmodel.pkl', 'rb')
tfidf_model = pickle.load(input)
input.close()
# print(document[0])
# kw = tfidf_model.transform(document)

variants = [
            # 'ssA',
            # 'ssB',
            # 'slA',
            # 'slB',
            # 'llA',
            'llB',
        ]
for var in variants:
    fs = [
            '../data/round1/%s/train.txt'%var,
            # '../dataB/round2/%s.txt'%var,
            # '../dataB/round3/divided_20210419/%s/train.txt'%var,
            # '../dataB/rematch/%s/train.txt'%var,
        ]
    for fname in fs: ##  = '../data/round1/%s/train.txt'%var
        texts = []
        labels = []
        with open(fname,encoding='utf-8') as f:
            lines = f.readlines()
            for l in tqdm(lines):
                l = l.strip()
                l = json.loads(l)
                texts.append(jieba.lcut(l['source']))
                texts.append(jieba.lcut(l['target']))
                labels.append(l['labelB'])

        document = [" ".join(sent0) for sent0 in texts]
        sparse_result = tfidf_model.transform(document)
        sort = np.argsort(sparse_result.toarray(),axis=1)[:,-300:]
        names = tfidf_model.get_feature_names()
        kw = pd.Index(names)[sort]                                            ### 获取关键字的主要程序

        print(len(labels))
        print(len(kw))
        assert len(kw)%2 == 0
        assert len(kw)//2 == len(labels)
        fname = fname[:7]+'B'+fname[7:]
        with open(fname,'w+',encoding='utf-8') as fout:
            for i in tqdm(range(0,len(document),2)):
                tmp = {}
                tmp['source'] = new_sent(document[i],kw[i])
                tmp['target'] = new_sent(document[i+1],kw[i+1])
                tmp['labelB'] = labels[i//2]
                fout.write(json.dumps(tmp,ensure_ascii=False)+'\n')


