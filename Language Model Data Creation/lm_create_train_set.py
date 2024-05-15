import json
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

METADATA_DIRECTORY = ''
DATA_DIRECTORY = ''
OUTPUT_DIRECTORY = ''

privaseer = set()
with open(METADATA_DIRECTORY) as f:
    data = f.readlines()
for line in data:
    line = json.loads(line)
    if line['overlap_entry'] == False:
        privaseer.add(line['hash'])
with open(METADATA_DIRECTORY) as f:
    data = f.readlines()
for line in data:
    line = json.loads(line)
    if line['overlap_entry'] == False:
        privaseer.add(line['file_hash'])

files = os.listdir(DATA_DIRECTORY)
file_count = 1

for file in files:
    text = []
    with open(DATA_DIRECTORY+file) as f:
        data = f.readlines()
    for line in data:
        line = json.loads(line)
        if line['hash'] in privaseer:
            text.append(line['text'])
    for i in range(len(text)):
        text[i] = text[i].split('\n')
        temp = []
        for j in text[i]:
            counter = 0
            l = len(j.split(' '))
            if l < 400 and l > 5:
                temp.append(j)
            elif l > 400:
                span = ''
                sentences = sent_tokenize(j)
                for sentence in sentences:
                    s_l = len(sentence.split(' '))
                    if s_l + counter > 400:
                        temp.append(span)
                        span = sentence
                        counter = s_l
                    else:
                        span = span + ' ' + sentence
                        counter += s_l
                temp.append(span)
        text[i] = temp
    with open(OUTPUT_DIRECTORY+str(file_count), 'w') as f:
        for doc in text:
            for line in doc:
                f.write(line+'\n')
    file_count += 1