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


train_data, valid_data,test_data = prepare('B')
train_loader = DataLoader(Mydata(train_data),batch_size=batch_size,shuffle=True,collate_fn=mycollate_a)
valid_loader = DataLoader(Mydata(valid_data),batch_size=batch_size,shuffle=True,collate_fn=mycollate_a)
test_loader = DataLoader(Mydata(test_data),batch_size=batch_size,shuffle=True,collate_fn=mycollate_a)

model = ESIM()
if use_gpu:
    model = model.cuda()

### 利用这两段函数进行BERT调节，将参与训练的层的requires_grad调节为True
### BERT层数在embedding过程中全部设置为requires_grad=False
# for i,(name,para) in enumerate(model.named_parameters()) :   ## 
#     if i > 180 :
#         para.requires_grad = True
# for i,(name,para) in enumerate(model.named_parameters()) :   ##
#     print(i,name,para.requires_grad)
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(
    filter(lambda p:p.requires_grad, model.parameters() ),
    lr = learning_rate
)
best_f1_a,best_f1_b = 0.,0.
for epoch in range(epochs):
    for i,data in enumerate(train_loader):
        # print(i)
        batch_flags , encoded_sents, batch_labels = data
        source_con,target_con = encoded_sents
        sr_batch_token_ids,sr_batch_segment_ids,sr_batch_attention_mask = source_con
        tg_batch_token_ids,tg_batch_segment_ids,tg_batch_attention_mask = target_con
        model.train()
        logits,p = model(sr_batch_token_ids,sr_batch_segment_ids,sr_batch_attention_mask,
                tg_batch_token_ids,tg_batch_segment_ids,tg_batch_attention_mask)
        loss = criterion(logits,batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 200 == 0:
            print('epoch_{}, loss值为：{}'.format(epoch+1,loss))
            evalA(model,[data])
    print('epoch_{}的验证集结果: '.format(epoch+1))
    f1_a,f1_b = evalA(model,valid_loader)
    if f1_b > best_f1_b:
        best_f1_b = f1_b
        torch.save(model.state_dict(),'BBbest_model.pkl')
    if f1_a > best_f1_a:
        best_f1_a = f1_a
        torch.save(model.state_dict(),'Abest_model.pkl')