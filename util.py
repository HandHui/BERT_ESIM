from transformers import AutoTokenizer
import torch
import re
import argparse
import json
import jieba
import numpy as np
import pandas as pd
import pickle


# input = open('dataHandle/tfmodel.pkl', 'rb')
# tfidf_model = pickle.load(input)
# input.close()

def get_parser():
    parser = argparse.ArgumentParser(description='Some Parameter')
    parser.add_argument('--use_gpu',default=1)
    parser.add_argument('--pre_train_model_path',default='model/chinese-roberta-wwm-ext/')
    parser.add_argument('--maxlen',default=512)
    parser.add_argument('--batch_size',default=16)
    parser.add_argument('--learning_rate',default=2e-5)
    parser.add_argument('--epochs',default=5)   ## A=5

    return parser


def clear_str(source):
    source = source.replace(' ','')
    source = re.sub('[\uAC00-\uD7AF]','',source)
    return source

'''
不过还是建议DataLoader()中的 collate_fn 参数自己去定义
下述collate族函数总有一些小问题(因为是仅适用于我自己的任务)
'''


def mycollate(datas):   ### 相似句作为前后两句传入的处理方法
    '''
    处理dataset(以batch为单位)
    return 返回的是批量的数据(按自己的需求完成数据处理)
    input: ...,source,target,label = data
    return: [CLS]+source+[SEP]+target+[SEP]
    '''
    parser = get_parser()
    parser = parser.parse_args()
    maxlen = parser.maxlen
    use_gpu = parser.use_gpu
    mpath = parser.pre_train_model_path
    tokenizer = AutoTokenizer.from_pretrained(mpath)

    batch_flags = []
    batch_token_ids, batch_segment_ids,batch_attention_mask = [],[],[]
    batch_labels = []
    for data in datas:
        ab_flag,source,target,label = data
        source = clear_str(source)
        target = clear_str(target)

        batch_flags.append(ab_flag)
        batch_labels.append(label)
        if len(source) > len(target):
            source,target = target,source
        
        if  len(target)>maxlen:
            source,target = get_key_word(tfidf_model,[source,target])

        
        ###处理token_ids,seg_ids,attention_mask(直接截断)
        if len(source)+len(target)>maxlen-3:  ###encode的过程便可以实现CLS/SEP的填充
            source = source[-maxlen+6:]                                                  ### 6这个数可以调节/只是为了确保不超过maxlen
            target = target[:maxlen-6]   ### 这种方法因为token.encode 与 len的长度不同
        # token_ids = tokenizer.encode(source,target)  encode会将 2014年 中的2014直接作为一个词
        source_ids = tokenizer.encode(source)
        target_ids = tokenizer.encode(target)

        if len(source_ids)+len(target_ids)-1>maxlen:  ###进行裁剪
            raw_len = len(source_ids)+len(target_ids)
            sp_len = (raw_len-maxlen+1)//2
            source_ids = source_ids[:-(sp_len+1)]+source_ids[-1:]
            target_ids = target_ids[:-(sp_len+1)]+target_ids[-1:]
            

        token_ids = source_ids+target_ids[1:]
        # if len(token_ids) > maxlen:
        #     print(source)
        #     print(target)
        #     assert 1==2
        # print(len(token_ids))
        assert len(token_ids) <= maxlen
        seg_ids = [0]*len(source_ids)+[1]*(len(target_ids)-1) 
        #         CLS +source +SEP        target+SEP
        
        assert len(token_ids) == len(seg_ids)
        attention_mask = [1]*len(token_ids)
        ## 填充
        token_ids = token_ids + (maxlen-len(token_ids))*[0]
        seg_ids = seg_ids + (maxlen-len(seg_ids))*[0]
        attention_mask = attention_mask + (maxlen-len(attention_mask))*[0]  ##先填充attention_mask
        
        ## 填入batch
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(seg_ids)
        batch_attention_mask.append(attention_mask)
        
    ## 转换为Tensor
    batch_flags = torch.tensor(batch_flags)
    batch_labels = torch.tensor(batch_labels)
    batch_token_ids = torch.tensor(batch_token_ids)
    batch_segment_ids = torch.tensor(batch_segment_ids)
    batch_attention_mask = torch.tensor(batch_attention_mask)
    if use_gpu:
        batch_flags = batch_flags.cuda()
        batch_labels = batch_labels.cuda()
        batch_token_ids = batch_token_ids.cuda()
        batch_segment_ids = batch_segment_ids.cuda()
        batch_attention_mask = batch_attention_mask.cuda()
    return (batch_flags,   batch_token_ids,batch_segment_ids,batch_attention_mask
           ,batch_labels)

#### 根据input_ids&maxlen  PADDING生成input_ids/segment_ids/attention_mask
def get_bert_inputs(input_ids):
    parser = get_parser()
    parser = parser.parse_args()
    maxlen = parser.maxlen
    seg_ids = [0]*len(input_ids)
    atten_mask = [1]*len(input_ids)
    input_ids = input_ids + (maxlen-len(input_ids))*[0]
    seg_ids = seg_ids + (maxlen-len(seg_ids))*[0]
    atten_mask = atten_mask + (maxlen-len(atten_mask))*[0]
    return input_ids,seg_ids,atten_mask


def mycollate_a(datas):   ### BERT_ESIM处理传进的相似句子
    '''
    input: ...,source,target,label = data 
    return: [CLS]+source+[SEP]
            [CLS]+target+[SEP] 
    '''

    parser = get_parser()
    parser = parser.parse_args()
    maxlen = parser.maxlen
    mpath = parser.pre_train_model_path
    use_gpu = parser.use_gpu
    tokenizer = AutoTokenizer.from_pretrained(mpath)
    
    batch_flags = []
    batch_labels = []
    # encoded_sents = []
    sr_batch_token_ids, sr_batch_segment_ids, sr_batch_attention_mask = [],[],[]
    tg_batch_token_ids, tg_batch_segment_ids, tg_batch_attention_mask = [],[],[]

    for data in datas:
        flag, source, target, label = data
        # if type(label[0]) != int:
        #     label = [0]*
        source = clear_str(source)
        target = clear_str(target)

        batch_flags.append(flag)
        batch_labels.append(label)
        if len(source)>len(target):
            source,target = target,source
        # if  len(target)>maxlen:
        #     source,target = get_key_word(tfidf_model,[source,target])
        
        if len(source)>maxlen-2:
            span = 250
            source = source[:span]+source[-span:]
        input_ids = tokenizer.encode(source)
        input_ids,seg_ids,atten_mask = get_bert_inputs(input_ids)
        if len(input_ids)>512:                                                    ####这就是我说的可能有问题的部分
            print('source')
            print(len(input_ids))
            print(source)
            print(target)
            assert 1==2
        sr_batch_token_ids.append(input_ids)
        sr_batch_segment_ids.append(seg_ids)
        sr_batch_attention_mask.append(atten_mask)
        
        if len(target)>maxlen-2:
            span = 250
            target = target[:span]+target[-span:]
        input_ids = tokenizer.encode(target)
        if len(input_ids)>512:                                                    #### 同样可能有问题
            print('target')
            print(source)
            print(target)
            print(len(target))
            print(span)
            assert 1==2
        input_ids,seg_ids,atten_mask = get_bert_inputs(input_ids)
        tg_batch_token_ids.append(input_ids)
        tg_batch_segment_ids.append(seg_ids)
        tg_batch_attention_mask.append(atten_mask)
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    batch_flags = torch.tensor(batch_flags).to(device)
    batch_labels = torch.tensor(batch_labels).to(device)
    sr_batch_token_ids = torch.tensor(sr_batch_token_ids).to(device)
    sr_batch_segment_ids = torch.tensor(sr_batch_segment_ids).to(device)
    sr_batch_attention_mask = torch.tensor(sr_batch_attention_mask).to(device)
    tg_batch_token_ids = torch.tensor(tg_batch_token_ids).to(device)
    tg_batch_segment_ids = torch.tensor(tg_batch_segment_ids).to(device)
    tg_batch_attention_mask = torch.tensor(tg_batch_attention_mask).to(device)
    return batch_flags, (
            (sr_batch_token_ids,sr_batch_segment_ids,sr_batch_attention_mask),
            (tg_batch_token_ids,tg_batch_segment_ids,tg_batch_attention_mask)
        ), batch_labels



def mycollate_test(datas):   ##BERT+ESIM结构的测试集输入，不同于上述内容，输入中不存在label

    '''
    input: ...,source,target,label = data 
    return: [CLS]+source+[SEP]
            [CLS]+target+[SEP] 
    '''

    parser = get_parser()
    parser = parser.parse_args()
    maxlen = parser.maxlen
    mpath = parser.pre_train_model_path
    use_gpu = parser.use_gpu
    tokenizer = AutoTokenizer.from_pretrained(mpath)
    
    batch_flags = []
    batch_ids = []
    # encoded_sents = []
    sr_batch_token_ids, sr_batch_segment_ids, sr_batch_attention_mask = [],[],[]
    tg_batch_token_ids, tg_batch_segment_ids, tg_batch_attention_mask = [],[],[]

    for data in datas:
        flag, source, target, ids = data
        
        source = clear_str(source)
        target = clear_str(target)

        batch_flags.append(flag)
        batch_ids.append(ids)
        if len(source)>len(target):
            source,target = target,source
        # if  len(target)>maxlen:
        #     source,target = get_key_word(tfidf_model,[source,target])
        
        if len(source)>maxlen-2:
            span = 250
            source = source[:span]+source[-span:]
        input_ids = tokenizer.encode(source)
        input_ids,seg_ids,atten_mask = get_bert_inputs(input_ids)
        if len(input_ids)>512:
            print('source')
            print(len(input_ids))
            print(source)
            print(target)
            assert 1==2
        sr_batch_token_ids.append(input_ids)
        sr_batch_segment_ids.append(seg_ids)
        sr_batch_attention_mask.append(atten_mask)
        
        if len(target)>maxlen-2:
            span = 250
            target = target[:span]+target[-span:]
        input_ids = tokenizer.encode(target)
        if len(input_ids)>512:                                                      ####可能有问题
            print('target')
            print(source)
            print(target)
            print(len(target))
            print(span)
            assert 1==2
        input_ids,seg_ids,atten_mask = get_bert_inputs(input_ids)
        tg_batch_token_ids.append(input_ids)
        tg_batch_segment_ids.append(seg_ids)
        tg_batch_attention_mask.append(atten_mask)
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    batch_flags = torch.tensor(batch_flags).to(device)
    # batch_labels = torch.tensor(batch_labels).to(device)
    sr_batch_token_ids = torch.tensor(sr_batch_token_ids).to(device)
    sr_batch_segment_ids = torch.tensor(sr_batch_segment_ids).to(device)
    sr_batch_attention_mask = torch.tensor(sr_batch_attention_mask).to(device)
    tg_batch_token_ids = torch.tensor(tg_batch_token_ids).to(device)
    tg_batch_segment_ids = torch.tensor(tg_batch_segment_ids).to(device)
    tg_batch_attention_mask = torch.tensor(tg_batch_attention_mask).to(device)
    return batch_flags, (
            (sr_batch_token_ids,sr_batch_segment_ids,sr_batch_attention_mask),
            (tg_batch_token_ids,tg_batch_segment_ids,tg_batch_attention_mask)
        ), batch_ids



def get_tp_fp_fn(pred,labels):      ### 获得tp/fp/tn个数
    
    tp = (pred*labels).sum()
    fp = ((pred-labels)>0).sum()
    fn = ((labels-pred)>0).sum()
    
    return tp,fp,fn

def evalr(model,data_loader,mode='train'):    ###自己任务的评价参数PRF
    tp_a,fp_a,fn_a = 0,0,0
    tp_b,fp_b,fn_b = 0,0,0
    # if mode == 'train':
    for data in data_loader:
        batch_flags, batch_token_ids,batch_segment_ids,batch_attention_mask,batch_labels = data
        
        label_logis = model(batch_token_ids,batch_segment_ids,batch_attention_mask)
        pred_labels = torch.softmax(label_logis,dim=-1).argmax(dim=-1)
        tp,fp,fn = get_tp_fp_fn(pred_labels,batch_labels)

        flag = batch_flags[0]  ## 判断A或者B
        if flag%2 == 0:
            
            tp_a += tp
            fp_a += fp
            fn_a += fn
        else: 
            tp_b += tp
            fp_b += fp
            fn_b += fn
    if flag%2 == 0:
        
        p = tp_a/(tp_a+fp_a+0.0000001)
        r = tp_a/(tp_a+fn_a+0.0000001)
        # if p == 0 and r == 0:
        #     with open('err.json','w+',encoding='utf-8') as f:
        #         json.dump({'con':batch_token_ids.tolist(),'prediction':pred_labels.tolist(),
        #         'label':batch_labels.tolist()},f,ensure_ascii=False,indent=4)
        print('A-->P:{},R:{},F:{}'.format(p,r,2*p*r/(p+r+0.0000001)))
    else:
        p = tp_b/(tp_b+fp_b)
        r = tp_b/(tp_b+fn_b)
        print('B-->P:{},R:{},F:{}'.format(p,r,2*p*r/(p+r)))

def eval(model,data_loader):    ###  自己任务的评价参数accuracy(纯BERT)
    model.eval()
    total_a, right_a = 0.,0.
    total_b, right_b = 0., 0.
    for data in data_loader:
        batch_flags, batch_token_ids,batch_segment_ids,batch_attention_mask, \
                    batch_labels = data
        label_logis = model(batch_token_ids,batch_segment_ids,batch_attention_mask)
        pred_labels = torch.softmax(label_logis,dim=-1).argmax(dim=-1)

        batch_flags = batch_flags%2
        total_b += batch_flags.sum()
        total_a += len(batch_flags)-total_b

        right_a += ((batch_labels == pred_labels)*(batch_flags==0)).sum()
        right_b += ((batch_labels == pred_labels)*(batch_flags==1)).sum()
    f1_a,f1_b = (right_a/(total_a+0.0000001)).item() , (right_b/(total_b+0.0000001)).item()
    print('f1_a:',f1_a,'f1_b:',f1_b,'f1:',(f1_a+f1_b)/2)
    return f1_a,f1_b


def evalA(model,data_loader):   ####(BERT_ESIM模型完成评价accuracy)
    total_a, right_a = 0.,0.
    total_b, right_b = 0., 0.

    for data in data_loader:
        batch_flags , encoded_sents, batch_labels = data
        source_con,target_con = encoded_sents
        sr_batch_token_ids,sr_batch_segment_ids,sr_batch_attention_mask = source_con
        tg_batch_token_ids,tg_batch_segment_ids,tg_batch_attention_mask = target_con

        logits,p = model(sr_batch_token_ids,sr_batch_segment_ids,sr_batch_attention_mask,
            tg_batch_token_ids,tg_batch_segment_ids,tg_batch_attention_mask)
        
        pred_labels = p.argmax(dim=-1)

        batch_flags = batch_flags%2
        total_b += batch_flags.sum()
        total_a += len(batch_flags)-total_b

        right_a += ((batch_labels == pred_labels)*(batch_flags==0)).sum()
        right_b += ((batch_labels == pred_labels)*(batch_flags==1)).sum()
    f1_a,f1_b = (right_a/(total_a+0.0000001)).item() , (right_b/(total_b+0.0000001)).item()
    print('f1_a:',f1_a,'f1_b:',f1_b,'f1:',(f1_a+f1_b)/2)
    return f1_a,f1_b

def get_k_fold(length):      ####五折交叉验证的验证集区间
    span = length//5
    res = []
    start = 0
    for i in range(4):
        res.append([start,start+span])
        start = start+span
    res.append([start,length])
    return res



### 下述两个函数完成利用sklearn.TFIDF模型抽取出关键字并按原顺序进行还原的任务
### 即按顺序在原句中删除非关键字
### tf_idf_model来自于最前端的那个被注释掉的全局变量
def new_sent(d0,k0):
    con_lst = []
    if len(d0)<510:
        return d0
    for c in jieba.lcut(d0):
        if c in k0:
            con_lst.append(c)
    return ''.join(con_lst)


def get_key_word(tfidf_model,texts):
    texts = [jieba.lcut(text) for text in texts]
    document = [" ".join(sent0) for sent0 in texts]
    sparse_result = tfidf_model.transform(document)
    sort = np.argsort(sparse_result.toarray(),axis=1)[:,-300:]
    names = tfidf_model.get_feature_names()
    kw = pd.Index(names)[sort]

    i = 0
    sr = new_sent(document[i],kw[i])
    tg = new_sent(document[i+1],kw[i+1])
    return (sr,tg)



