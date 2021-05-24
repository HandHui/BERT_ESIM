import argparse
import torch.nn as nn
from model.ESIM import ESIM
from dataHandle.dataPrepare import prepare
from dataHandle.MyLoader import Mydata
import torch
from torch.utils.data import Dataset,DataLoader
from util import *

parser = get_parser()
parser = parser.parse_args()
batch_size = parser.batch_size
use_gpu = parser.use_gpu
learning_rate = parser.learning_rate
epochs = parser.epochs

AB = [
    ['A','Abest_model.pkl','Aresult.csv'],
    ['B','Bbest_model.pkl','Bresult.csv']
]

for ab in AB:

    _, _,test_data = prepare(ab[0])
    # train_loader = DataLoader(Mydata(train_data),batch_size=batch_size,shuffle=True,collate_fn=mycollate_a)
    # valid_loader = DataLoader(Mydata(valid_data),batch_size=batch_size,shuffle=True,collate_fn=mycollate_a)
    test_loader = DataLoader(Mydata(test_data),batch_size=batch_size,collate_fn=mycollate_test)

    model = ESIM()
    model.load_state_dict(torch.load(ab[1]))
    if use_gpu:
        model = model.cuda()
    model.eval()
    with open(ab[2],'w+') as fout:
        fout.write('id,label\n')
        for i,data in enumerate(test_loader):
            print(i)
            batch_flags , encoded_sents, batch_ids = data
            source_con,target_con = encoded_sents
            sr_batch_token_ids,sr_batch_segment_ids,sr_batch_attention_mask = source_con
            tg_batch_token_ids,tg_batch_segment_ids,tg_batch_attention_mask = target_con
            
            logits,p = model(sr_batch_token_ids,sr_batch_segment_ids,sr_batch_attention_mask,
                    tg_batch_token_ids,tg_batch_segment_ids,tg_batch_attention_mask)
            p = p.argmax(dim=-1)
            # p = p.to(torch.device('cpu'))
            for idx,y in zip(batch_ids,p):
                # print(idx)
                # print(y.item())
                fout.write('%s,%s\n' % (idx, y.item()))

            # print(batch_ids)
            # print(p)
            # break
      