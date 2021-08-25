import json
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers import BertTokenizer
from random import choice
import codecs

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
        # model_name = "bert-base-multilingual-cased"
        model_name = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text, gold = self.data[item]['text'], self.data[item]['spo_list']
        text = text if len(text) <= 512 else text[:512]
        sample = list(text)
        sample = self.tokenizer.convert_tokens_to_ids(sample)
        sub_start = len(sample)*[0]
        sub_end = len(sample)*[0]
        relation_start = [[0 for _ in range(self.config['relation_types'])] for _ in range(len(sample))]
        relation_end = [[0 for _ in range(self.config['relation_types'])] for _ in range(len(sample))]
        #   dim = (seq_len, relation_types)
        sub_start_single = len(sample) * [0]
        sub_end_single = len(sample)*[0]
        s2ro_map = {}
        for entry in gold:
            sub = entry['subject']
            obj = entry['object']['@value']
            relation = '同义词-' + entry['subject_type'] if entry['predicate'] == '同义词' else entry['predicate']
            #正则表达式无法处理小括号, 所以抛出异常
            try:
                sub_pos = re.search(sub, text).span()
                obj_pos = re.search(obj, text).span()
                relation_idx = self.relation2idx[relation]
                sub_start[sub_pos[0]] = 1
                sub_end[sub_pos[1]-1] = 1
                if sub_pos not in s2ro_map:
                    s2ro_map[sub_pos] = []
                s2ro_map[sub_pos].append((obj_pos, relation_idx))
            except:
                pass
        if s2ro_map:
            sub_pos = choice(list(s2ro_map.keys()))
            sub_start_single[sub_pos[0]] = 1
            sub_end_single[sub_pos[1] - 1] = 1
            for obj_pos, relation_idx in s2ro_map.get(sub_pos, []):
                relation_start[obj_pos[0]][relation_idx] = 1
                relation_end[obj_pos[1]-1][relation_idx] = 1
        return sample, sub_start, sub_end, relation_start, relation_end, sub_start_single, sub_end_single

def collate_fn(data):
    data.sort(key= lambda x: len(x[0]), reverse = True)
    sample, sub_start, sub_end, relation_start, relation_end, sub_start_single, sub_end_single = zip(*data)
    mask = [[1 if j < len(i) else 0 for j in range(len(sample[0]))] for i in sample]
    sample = [torch.tensor(i).long().to(device) for i in sample]
    sub_start = [torch.tensor(i).long().to(device) for i in sub_start]
    sub_end = [torch.tensor(i).long().to(device) for i in sub_end]
    relation_start = [torch.tensor(i).long().to(device) for i in relation_start]
    relation_end = [torch.tensor(i).long().to(device) for i in relation_end]
    sub_start_single = [torch.tensor(i).long().to(device) for i in sub_start_single]
    sub_end_single = [torch.tensor(i).long().to(device) for i in sub_end_single]
    mask = torch.tensor(mask).long().to(device)
    sample = pad_sequence(sample, batch_first=True, padding_value=0)
    sub_start = pad_sequence(sub_start, batch_first=True, padding_value=0)
    sub_end = pad_sequence(sub_end, batch_first=True, padding_value=0)
    relation_start = pad_sequence(relation_start, batch_first=True, padding_value=0)
    relation_end = pad_sequence(relation_end, batch_first=True, padding_value=0)
    sub_start_single = pad_sequence(sub_start_single, batch_first=True, padding_value=0)
    sub_end_single = pad_sequence(sub_end_single, batch_first=True, padding_value=0)
    return sample, sub_start, sub_end, relation_start, relation_end, mask, sub_start_single, sub_end_single
#   dim(sample) = dim(sub_start) = dim(sub_end) = (batch_size, seq_len]
#   dim(relation_start) = dim(relation_end) = (batch_size, seq_len, relation_types)


if __name__ == "__main__":
    path = 'data/CMeIE_train.json'
    config = {"mode": "train", "relation_types": 53}
    data = MyDataset(path, config)
    dataloader = DataLoader(data, batch_size=40, shuffle=False, collate_fn=collate_fn)
    batch_data = next(iter(dataloader))

    file = codecs.open('debug.txt', 'w', encoding='utf-8')

    test_idx = 37
    a, b = batch_data[3][test_idx], batch_data[4][test_idx]
    for i in batch_data:
        file.write(str(i[test_idx])+'\n')
    for i in range(a.shape[0]):
        file.write(str(i)+str(b[i])+'\n')
