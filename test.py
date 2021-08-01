import codecs

import torch
import torch.nn.functional as F
from casrel import CasRel
from torch.utils.data import DataLoader
from dataloader import MyDataset, collate_fn
import json

device = 'cuda:0'
torch.set_num_threads(6)

def trans_schemas(path):
    re2sub = dict()
    re2obj = dict()
    with open(path, "r", encoding='utf-8') as f:
        sens = f.readlines()
    schemas = []
    for sen in sens:
        schemas.append(json.loads(sen.strip()))
    for entry in schemas:
        re2sub[entry['predicate']] = entry['subject_type']
        re2obj[entry['predicate']] = entry['object_type']
    return re2sub, re2obj

def get_list(mode, start, end, text):
    if(mode == 'sub'):
        threshold = 0.5
    elif(mode == 'obj'):
        threshold = 0.5
    res = []
    n = min(len(text), 512)
    i, j = 0, 0
    while(i < n and j < n):
        while(i < n and start[i] < threshold):
            i += 1
        while(j < n and end[j] < threshold):
            j += 1
        if(i < n and j < n and i <= j):
            entry = dict()
            entry['text'] = text[i : j+1]
            entry['start'] = i
            entry['end'] = j
            res.append(entry)
        elif(i < n and j < n and i > j):
            i -= 1
        i += 1
        j += 1
    return res

def get_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [json.loads(i) for i in data]
    return data

def get_real_index(texts):
    zip_texts = list(zip(texts, list(range(len(texts)))))
    zip_texts.sort(key = lambda x : len(x[0]), reverse=True)
    real_idx = dict()
    for i in range(len(zip_texts)):
        real_idx[zip_texts[i][1]] = i
    return real_idx

def match(sub_list, obj_list, relation_type, text, re2sub, re2obj):
    spo_list = []
    for sub in sub_list:
        for obj in obj_list:
            entry = dict()
            entry['Combined'] = '。' in text[sub['end']: obj['start']] or '。' in text[obj['end']: sub['start']]
            entry['predicate'] = relation_type
            entry['subject'] = sub['text']
            entry['subject_type'] = re2sub[relation_type]
            entry['object'] = {'@value': obj['text']}
            entry['object_type'] = {'@value': re2obj[relation_type]}
            spo_list.append(entry)
    return spo_list
#   完全匹配

if __name__ == '__main__':
    config = {'mode': 'test', 'batch_size': 83, 'epoch': 40, 'relation_types': 53}
    path = 'data/CMeIE_train.json'
    schemas_path = 'data/53_schemas.json'
    res_path = 'data/CMeIE_test_res.json'
    res_file = codecs.open(res_path, 'w', encoding='utf-8')
    raw_data = get_text(path)
    re2sub, re2obj = trans_schemas(schemas_path)
    data = MyDataset(path, config)
    dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    model = CasRel(config).to(device)
    model.load_state_dict(torch.load('params_mul.pkl'))
    res = []
    for batch_index, (sample, sub_start, sub_end, relation_start, relation_end, mask) in enumerate(iter(dataloader)):
        batch_data = dict()
        batch_data['token_ids'] = sample
        batch_data['mask'] = mask
        texts = [raw_data[i]['text'] for i in range(batch_index*config['batch_size'], (batch_index+1)*config['batch_size'])]
        real_idx = get_real_index(texts)    #   index_before_sort --> index_after_sort
        pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end = model(batch_data)
        pred_sub_start = pred_sub_start.squeeze(-1) #   (batch_size, seq_len, 1) --> (batch_size, seq_len)
        pred_sub_end = pred_sub_end.squeeze(-1)
        pred_obj_start = pred_obj_start.transpose(1, 2) #   (batch_size, seq_len, relation_types) --> (batch_size, relation_types, seq_len)
        pred_obj_end = pred_obj_end.transpose(1, 2)
        for i in range(config['batch_size']):
            sub_list = get_list('sub', pred_sub_start[real_idx[i]], pred_sub_end[real_idx[i]], raw_data[batch_index*config['batch_size']+i]['text'])
            spo_list = []
            for j in range(config['relation_types']):
                obj_list = get_list('obj', pred_obj_start[real_idx[i]][j], pred_obj_end[real_idx[i]][j], raw_data[batch_index*config['batch_size']+i]['text'])
                spo_list += match(sub_list, obj_list, data.idx2relation[j], raw_data[batch_index*config['batch_size']+i]['text'], re2sub, re2obj)
            entry = dict()
            entry['text'] = raw_data[batch_index*config['batch_size']+i]['text']
            entry['spo_list'] = spo_list
            json.dump(entry, res_file, ensure_ascii=False)
            res_file.write('\n')
        print("batch: %d"% batch_index)