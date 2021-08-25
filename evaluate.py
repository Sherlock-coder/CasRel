import json
import codecs
from test import test
from config import *

def evaluate():
    pred_path = "data/CMeIE_dev_res.json"   # prediction file
    gold_path = "data/CMeIE_dev.json"   # gold file
    res_path = 'eval_dev.json'
    pred_file = codecs.open(pred_path, 'r', encoding='utf-8')
    gold_file = codecs.open(gold_path, 'r', encoding='utf-8')
    res_file = codecs.open(res_path, 'w', encoding='utf-8')
    pred_data = pred_file.readlines()
    pred_data = [json.loads(i) for i in pred_data]
    gold_data = gold_file.readlines()
    gold_data = [json.loads(i) for i in gold_data]
    data = zip(pred_data, gold_data)
    correct_num = 0
    pred_num = 0
    gold_num = 0
    for pred, gold in data:
       pred_spo = [(i['predicate'], i['subject'], i['subject_type'], i['object']['@value'], i['object_type']['@value']) for i in pred['spo_list']]
       pred_spo = set(pred_spo)
       gold_spo = [(i['predicate'], i['subject'], i['subject_type'], i['object']['@value'], i['object_type']['@value']) for i in gold['spo_list']]
       gold_spo = set(gold_spo)
       correct_num +=  len(pred_spo & gold_spo)
       pred_num += len(pred_spo)
       gold_num += len(gold_spo)
       entry = {}
       entry['text'] = pred['text']
       entry['gold'] = list(gold_spo)
       entry['pred'] = list(pred_spo)
       entry['new'] = list(pred_spo - gold_spo)
       entry['lack'] = list(gold_spo - pred_spo)
       json.dump(entry, res_file, ensure_ascii=False, indent=4, separators=(',', ':'))
    eps = 1e-6
    p = correct_num / (pred_num + eps)
    r = correct_num / (gold_num + eps)
    f1 = (2 * p * r) / (p + r + eps)
    print("p:%f r:%f f1:%f"%(p, r, f1))
    return f1

if __name__ == '__main__':
    evaluate()
