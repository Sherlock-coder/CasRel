import codecs

import torch
import torch.nn.functional as F
from casrel import CasRel
from torch.utils.data import DataLoader
from dataloader import MyDataset, collate_fn
import json
import numpy as np

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


def get_list(start, end, text, h_bar=0.5, t_bar=0.5):
    res = []
    start, end = start[: 512], end[: 512]
    start_idxs, end_idxs = [], []
    for idx in range(len(start)):
        if (start[idx] > h_bar):
            start_idxs.append(idx)
        if (end[idx] > t_bar):
            end_idxs.append(idx)
    for start_idx in start_idxs:
        for end_idx in end_idxs:
            if (end_idx >= start_idx):
                entry = {}
                entry['text'] = text[start_idx: end_idx+1]
                entry['start'] = start_idx
                entry['end'] = end_idx
                res.append(entry)
                break
    return res

def get_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [json.loads(i) for i in data]
    return data


if __name__ == '__main__':
    config = {'mode': 'test', 'batch_size': 1, 'relation_types': 53}
    path = 'data/CMeIE_test.json'
    schemas_path = 'data/53_schemas.json'
    res_path = 'data/CMeIE_test_res.json'
    res_file = codecs.open(res_path, 'w', encoding='utf-8')
    raw_data = get_text(path)
    re2sub, re2obj = trans_schemas(schemas_path)
    data = MyDataset(path, config)
    dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    model = CasRel(config).to(device)
    model.load_state_dict(torch.load('params.pkl'))
    for batch_index, (sample, sub_start, sub_end, relation_start, relation_end, mask, _, _) in enumerate(iter(dataloader)):
        with torch.no_grad():
            text =  raw_data[batch_index]['text']
            batch_data = dict()
            batch_data['token_ids'] = sample
            batch_data['mask'] = mask
            batch_data['text'] = text
            ret = model.test(batch_data)
            spo_list = []
            if ret:
                sub_list, pred_obj_start, pred_obj_end = ret
                for idx, sub in enumerate(sub_list):
                    obj_start, obj_end = pred_obj_start[idx].transpose(0, 1), pred_obj_end[idx].transpose(0, 1)
                    for i in range(config['relation_types']):
                        obj_list = get_list(obj_start[i], obj_end[i], text)
                        for obj in obj_list:
                            entry = {}
                            entry['Combined'] = '。' in text[sub['end']: obj['start']] or '。' in text[obj['end']: sub['start']]
                            entry['subject'] = sub['text']
                            entry['predicate'] = data.idx2relation[i]
                            entry['object'] = {'@value': obj['text']}
                            entry['subject_type'] = re2sub[data.idx2relation[i]]
                            entry['object_type'] = {'@value': re2obj[data.idx2relation[i]]}
                            spo_list.append(entry)
            res = {}
            res['text'] = text
            res['spo_list'] = spo_list
            json.dump(res, res_file, ensure_ascii=False)
            res_file.write('\n')
            print(batch_index)