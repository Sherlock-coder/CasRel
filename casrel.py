import torch
import torch as t
from torch import nn
from transformers import BertModel

class CasRel(nn.Module):
    def __init__(self, config):
        super(CasRel, self).__init__()
        self.config = config
        self.bert_dim = 768
        model_name = 'bert-base-chinese'
        # model_name = "bert-base-multilingual-cased"
        self.bert_encoder = BertModel.from_pretrained(model_name)
        self.sub_start_tageer = nn.Linear(self.bert_dim, 1)
        self.sub_end_tagger = nn.Linear(self.bert_dim, 1)
        self.obj_start_tagger = nn.Linear(self.bert_dim, config['relation_types'])
        self.obj_end_tagger = nn.Linear(self.bert_dim, config['relation_types'])
        self.sub_threshold = 0.5

    def get_encoded_text(self, data):
        with torch.no_grad():   #   out of GPU Memory
            encoded_text = self.bert_encoder(data['token_ids'], attention_mask=data['mask'])[0]
        return encoded_text #   dim  = (batch_size, seq_len, bert_dim)

    def get_sub(self, encoded_text):
        #   dim(pred) = (batch_size, seq_len, 1)
        pred_sub_start = self.sub_start_tageer(encoded_text)
        pred_sub_start = torch.sigmoid(pred_sub_start)
        pred_sub_end = self.sub_end_tagger(encoded_text)
        pred_sub_end = torch.sigmoid(pred_sub_end)
        return pred_sub_start, pred_sub_end

    # def get_sub_info(self, encoded_text, pred_sub_start, pred_sub_end, real_sub_start, real_sub_end):
    #     if(self.config['mode'] == 'train'):
    #         start = real_sub_start
    #         end = real_sub_end
    #     elif(self.config['mode'] == 'test'):
    #         threshold = 0.5
    #         start = [pred_sub_start > threshold]
    #         end = [pred_sub_end > threshold]
    #     pred_sub_lst = []
    #     for idx_start, i in enumerate(start):
    #         if(i == 1):
    #             for idx_end in range(idx_start, len(end)):
    #                 if(end[idx_end] == 1):
    #    Problem: 一个batch中不同样本的sub_list长度不同

    def get_obj(self, sub_start_mapping, sub_end_mapping, encoded_text):
        #   dim(sub_start_mapping) = dim(sub_end_mapping) = (batch_size, 1, seq_len)
        #   dim(encoded_text) = (batch_size, seq_len, bert_dim)
        sub_start = torch.matmul(sub_start_mapping.float(), encoded_text)
        sub_end = torch.matmul(sub_end_mapping.float(), encoded_text)
        #   dim(sub_start) = dim(sub_end) = (batch_size, 1, bert_dim)
        sub = (sub_start + sub_end) / 2
        encoded_text = encoded_text + sub
        pred_obj_start = self.obj_start_tagger(encoded_text)
        pred_obj_end = self.obj_end_tagger(encoded_text)
        pred_obj_start = torch.sigmoid(pred_obj_start)
        pred_obj_end = torch.sigmoid(pred_obj_end)
        return pred_obj_start, pred_obj_end
        #   dim = (batch_size, seq_len, relation_types)

    def forward(self, data):
        encoded_text = self.get_encoded_text(data)
        pred_sub_start, pred_sub_end = self.get_sub(encoded_text)
        if(self.config['mode'] == 'train'):
            sub_start_mapping = data['sub_start'].unsqueeze(1)
            # (batch_size, seq_len) --> (batch_size, 1, seq_len)
            sub_end_mapping = data['sub_end'].unsqueeze(1)
        elif(self.config['mode'] == 'test'):
            sub_start_mapping = (pred_sub_start.transpose(1, 2) >= self.sub_threshold).float()
            sub_end_mapping = (pred_sub_end.transpose(1, 2) >= self.sub_threshold).float()
        pred_obj_start, pred_obj_end = self.get_obj(sub_start_mapping, sub_end_mapping, encoded_text)
        return pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end
        #   dim(pred_sub_start) = dim(pred_sub_end) = (batch_size, seq_len, 1)
        #   dim(pred_obj_start) = dim(pred_obj_end) = (batch_size, seq_len, realtion_types)