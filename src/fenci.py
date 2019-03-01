import re
import psycopg2
import jieba
import os
from collections import Counter
import numpy as np
jieba.load_userdict("./data/800W-jiebauserdict.txt")
def fenju(data):
    sentence=[]
    label=[]
    for i in range(len(data)):
        try:
            m = re.findall('。',data[i][0])
            # print(m)
            if data[i][1] is not None and len(m)>0:
                if len(m)>1:
                    content=data[i][0].split('。')
                    # print(content)
                    for c in range(len(content)):
                        if len(content[c])>10:
                            sentence.append(content[c]+'。')
                            label.append(data[i][1])
                elif len(data[i][0])>10:
                    sentence.append(data[i][0])
                    label.append(data[i][1])
            else:
                continue
        except:
            continue
    assert (len(sentence) == len(label))
    return sentence, label

def _process_sentence_list(sentence_list, label, threshold=0.01):
    sentence_count = Counter(sentence_list)
    total_count = len(sentence_list)
    # 计算句子频率
    sentence_freqs = {w: c / total_count for w, c in sentence_count.items()}
    # 计算被删除的概率
    # prob_drop = {w: 1 - np.sqrt(t / sentence_freqs[w]) for w in sentence_count}
    # print(prob_drop)
    # 剔除出现频率太高的句子
    sentence=[]
    labels=[]
    for w in range(len(sentence_list)):
        if sentence_freqs[sentence_list[w]] < threshold:
            sentence.append(sentence_list[w])
            labels.append(label[w])
        else:
            continue
    # sentence_list = [w for w in sentence_list if sentence_freqs[w] < threshold]
    assert (len(sentence) == len(labels))
    return sentence, labels, total_count-len(sentence)

def fenci(alltext,label,writefile,filename,labelname):
    if not os.path.exists(writefile):
        os.makedirs(writefile)
    sentence = [' '.join(jieba.lcut(''.join(text.split()))) for text in alltext]
    print(sentence)
    with open(os.path.join(writefile, filename), "w") as fw:
        fw.write("\n".join(sentence))
    with open(os.path.join(writefile, labelname), "w") as fw1:
        fw1.write("\n".join(label))
#

def getdata(filedir, fencifile, labelfile):
    conn = psycopg2.connect(database='AIDB',user='tatt', password='123456', host='192.168.91.13', port='5432')
    cur = conn.cursor()
    cur.execute("select jslcm, ay from judgement_sh limit 50000")
    data = cur.fetchall()
    sentence, label = fenju(data)
    sentences,labels, delete = _process_sentence_list(sentence, label)
    print(delete)
    fenci(sentences, labels, filedir, fencifile, labelfile)
    # for i in range(len(data)):
    #     if data[i][0]:
    #        f.write(data[i][0]+'\n')
    # #     else:
    #         continue

getdata('./data/cpws', 'tokenized.txt', 'label.txt')