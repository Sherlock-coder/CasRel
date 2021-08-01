import json
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers import BertTokenizer

device = 'cuda:0'

class MyDataset(Dataset):
    def __get_train_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        res = [json.loads(i) for i in data]
        return res

    def __get_test_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        data = [json.loads(i) for i in data]
        res = []
        for entry in data:
            entry['spo_list'] = []
            res.append(entry)
        return res

    def __init__(self, path, config):
        super(MyDataset, self).__init__()
        self.config =config
        if(self.config['mode'] == 'train'):
            self.data = self.__get_train_data(path)
        elif(self.config['mode'] == 'test'):
            self.data = self.__get_test_data(path)
        with open('data/relation2idx.json', 'r', encoding='utf-8') as f:
            self.relation2idx = json.load(f)
        self.idx2relation = dict()
        for key in self.relation2idx:
            self.idx2relation[self.relation2idx[key]] = key
        model_name = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text, gold = self.data[item]['text'], self.data[item]['spo_list']
        text = text if len(text) <= 512 else text[:512]
        sample = list(text)
        sample = self.tokenizer.convert_tokens_to_ids(sample)
        sub_start = [0 for _ in range(len(sample))]
        sub_end = [0 for _ in range(len(sample))]
        relation_start = [[0 for _ in range(self.config['relation_types'])] for _ in range(len(sample))]
        relation_end = [[0 for _ in range(self.config['relation_types'])] for _ in range(len(sample))]
        #   dim = (seq_len, relation_types)
        for entry in gold:
            sub = entry['subject']
            obj = entry['object']['@value']
            relation = entry['predicate']
            #正则表达式无法处理小括号, 所以抛出异常
            try:
                sub_pos = re.search(sub, text).span()
                obj_pos = re.search(obj, text).span()
                relation_idx = self.relation2idx[relation]
                sub_start[sub_pos[0]] = 1
                sub_end[sub_pos[1]-1] = 1
                relation_start[obj_pos[0]][relation_idx] = 1
                relation_end[obj_pos[1]-1][relation_idx] = 1
            except:
                pass
        return sample, sub_start, sub_end, relation_start, relation_end

def collate_fn(data):
    data.sort(key= lambda x: len(x[0]), reverse = True)
    sample, sub_start, sub_end, relation_start, relation_end = zip(*data)
    mask = [[1 if j < len(i) else 0 for j in range(len(sample[0]))] for i in sample]
    sample = [torch.tensor(i).long().to(device) for i in sample]
    sub_start = [torch.tensor(i).long().to(device) for i in sub_start]
    sub_end = [torch.tensor(i).long().to(device) for i in sub_end]
    relation_start = [torch.tensor(i).long().to(device) for i in relation_start]
    relation_end = [torch.tensor(i).long().to(device) for i in relation_end]
    mask = torch.tensor(mask).long().to(device)
    sample = pad_sequence(sample, batch_first=True, padding_value=0)
    sub_start = pad_sequence(sub_start, batch_first=True, padding_value=0)
    sub_end = pad_sequence(sub_end, batch_first=True, padding_value=0)
    relation_start = pad_sequence(relation_start, batch_first=True, padding_value=0)
    relation_end = pad_sequence(relation_end, batch_first=True, padding_value=0)
    return sample, sub_start, sub_end, relation_start, relation_end, mask
#   dim(sample) = dim(sub_start) = dim(sub_end) = (batch_size, seq_len]
#   dim(relation_start) = dim(relation_end) = (batch_size, seq_len, relation_types)

if __name__ == "__main__":
    path = 'data/CMeIE_train.json'
    config = {"mode": "train", "relation_types": 53}
    data = MyDataset(path, config)
    dataloader = DataLoader(data, batch_size=40, shuffle=False, collate_fn=collate_fn)
    sample, sub_start, sub_end, relation_start, relation_end, mask = next(iter(dataloader))
    print(sample)