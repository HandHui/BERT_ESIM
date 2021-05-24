import torch
import torch.nn as nn
from transformers import AutoModel


###纯BERT模型
class SSModel(nn.Module):

    def __init__(self,bertpath,hidden_size=768,out_size=2,dropout_rate=0.5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bertpath)
        self.fc = nn.Linear(hidden_size,out_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self,token_ids,segment_ids,attention_mask):
        bout = self.bert(token_ids,segment_ids,attention_mask)
        cls = bout.pooler_output
        # print(cls.size())   ##[batch,768]
        # seq_vec = bout.last_hidden_state  ##[batch,seq_len,768]
        # print(seq_vec.size())
        out = self.fc(cls)
        out = self.dropout(out)
        return out  ## nn.CrossEntropyLoss要求输入不经过softmax

        # return torch.softmax(out,dim=-1)  ## 保证处理的是2维的数据