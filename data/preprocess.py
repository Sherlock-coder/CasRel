import json

path = "53_schemas.json"
with open(path, 'r', encoding='utf-8') as f:
    data = f.readlines()
lst = [json.loads(i.strip()) for i in data]
relation2idx = dict()
for idx, entry in enumerate(lst):
    relation2idx[entry['predicate']] = idx
print("ralation_types: %d"% len(relation2idx))
with open('relation2idx.json', 'w', encoding='utf-8') as f:
    json.dump(relation2idx, f, ensure_ascii=False)
