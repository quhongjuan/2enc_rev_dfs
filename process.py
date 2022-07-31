#coding=utf-8

import json
from transformers import AutoTokenizer

def load():
    with open('/mnt/xk/datasets/1-encoder/eval/src-dfs-tokening-d.txt', 'r') as f:
        descs = f.readlines()
    f.close()
    with open('/mnt/xk/datasets/1-encoder/eval/src-val-d.txt','r') as f:
        codes = f.readlines()
    f.close()
   # with open('/mnt/xk/datasets/1-encoder1/train/tgt-train-d.txt','r') as f:
    #    comments = f.readlines()
    #f.close()

    jsons= []
    
    for i in range(len(codes)):
        sample = {}
        sample["url"] = str(i)

        sample['code'] = codes[i].strip()
        sample['docstring'] = descs[i]
        sample['func_name'] = ""
        json_test = json.dumps(sample)
        jsons.append(json_test+"\n")

    with open('nvalid.json', 'w+', encoding='utf-8') as f:
        f.writelines(jsons)
    f.close()

def load2():
    with open('s_code.txt', 'r') as f:
        train_codes = f.readlines()
    f.close()
    with open('s_desc.txt', 'r') as f:
        train_desc = f.readlines()
    f.close()
    jsons = []
    for i in range(len(train_codes)):
        sample = {}
        sample['url'] = str(i)
        sample['code'] = train_codes[i]
        sample['docstring'] = train_desc[i]
        sample['func_name'] = str(i)
        json_text = json.dumps(sample)
        jsons.append(json_text + "\n")
    
    with open('s_net.json', 'w+', encoding='utf-8') as f:
        f.writelines(jsons)
    f.close()


# 8:1:1划分数据集
def load1(path):

    with open(path, 'r') as f:
        lines = f.readlines()
    f.close()
    #lines = lines[11000:12000]
    train_data = []
    val_data = []
    test_data = []
    lens = len(lines)
    for i in range(lens):
        #if i >= 0 and i <= lens * 0.8:
        line = json.loads(lines[i])
        #line['url'] = str(i)
        if i >= 0 and i < lens * 0.8:
            train_data.append(json.dumps(line) + "\n")
        elif i >= lens * 0.8 and i < lens * 0.9:
            val_data.append(json.dumps(line) + "\n")
        elif i >= lens * 0.9 and i < lens:
            test_data.append(json.dumps(line) + "\n")
    with open('s_train.json', 'w+', encoding='utf-8') as f:
        f.writelines(train_data)
    f.close()

    with open('s_val.json', 'w+', encoding='utf-8') as f:
        f.writelines(val_data)
    f.close()

    with open('s_test.json', "w+", encoding='utf-8') as f:
        f.writelines(test_data)
    f.close()
    
    #js = []
    #for line in lines:
    #    js.append(json.loads(line)) # 转json
import re
import numpy as np

# build train/valid dataset and balance pos and neg
# codebert train/valid 格式：(label, url, class.method, docstring, code) <CODESPLIT>
# our：(label, index, methodname, docstring, code), index是数据在原始数据集中的索引,
def build(data_path, name):
    new_datas = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
    f.close()
    js = []
    for line in lines:
       js.append(json.loads(line)) # 转json
    length = len(js)
    d = {}
    for ix, item in enumerate(js):
        print(ix)
        pos_index = item['url']
        pos_methodname = item['func_name']
        pos_code = tokening(format_str(item['code']))
        pos_docstring = tokening(format_str(item['docstring']))
        pos = (str(1), pos_index, pos_methodname, pos_docstring, pos_code)
        new_datas.append('<CODESPLIT>'.join(pos) + "\n")
        while True:
            neg_ix = np.random.randint(0, length)
            if neg_ix != ix: # 随机选一个做为负样本
                break
        neg_index = pos_index+'_'+js[neg_ix]['url']
        neg_methodname = js[neg_ix]['func_name']
        neg_code = tokening(format_str(js[neg_ix]['code']))
        #neg_docstring = tokening(format_str(js[neg_ix]['docstring']))
        neg = (str(0), neg_index, neg_methodname, pos_docstring, neg_code)
        new_datas.append('<CODESPLIT>'.join(neg) + "\n")
    np.random.seed(0)
    idxs = np.arange(len(new_datas))
    new_datas = np.array(new_datas, dtype=np.object)
    np.random.shuffle(idxs)
    new_datas = new_datas[idxs]
    with open(name + '.txt', 'w+') as f:
        f.writelines(new_datas)
    f.close()

# fenci
def tokening(string):
    new_line_list = re.findall(r"[\w']+|[^\>\<\,\\\{\}\(\)\[\]\w\s\;\:\?\"']+|[\>\<\?\,\{\}\(\)\\\[\]\;\:\"]",
                         string.strip())
    return " ".join(new_line_list)

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string



if __name__ == "__main__":
    # sample first 10000
    #load1("/mnt/xk/search/GitData/_2018train.json")
    #tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    #ss = "private void validateCreateTable(ConnectorTableMetadata meta){\n  validateColumns(meta);\n  validateLocalityGroups(meta);\n  if (!AccumuloTableProperties.isExternal(meta.getProperties())) {\n    validateInternalTable(meta);\n  }\n}"
    #ss = tokening(format_str(ss))
    #print(tokenizer.tokenize("hell \"jj wold\" "))
    #tokens = tokenizer.tokenize(' '.join([s.strip() for s in format_str(ss).split(' ')]))
    #ss = "1 2 3 4"
    #print(tokens)
    #print(tokening(format_str(ss)))
    #print()
    #print(tokens)
    #load2()
    #load1('s_net.json')
    load()
    #build("s_val.json", "ps_valid")
    # with open("1.txt", 'w+') as f:
    #     #f.writeline(' '.join(["1", "2"]))
    #     f.write(' '.join(tokens))
