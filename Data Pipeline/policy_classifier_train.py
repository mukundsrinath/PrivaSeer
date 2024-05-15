from bs4 import BeautifulSoup
from bs4.element import Comment
import pandas as pd
import json
from urllib.parse import unquote
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words
import nltk
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import nltk
import re, unicodedata
from nltk.corpus import stopwords
from tqdm import tqdm
from joblib import dump
import pickle
from sklearn.metrics import classification_report
from boilerpipe.extract import Extractor


'''

This file trains a random classifier given a list of candidate privacy policies and their labels. The labels are binary.

'''

directory = ''
metadata_directory = ''

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def get_text(i):
    text = ''
    if '.html' in i:
        with open(i, encoding='utf-8') as f:
            data = f.read()
        extractor = Extractor(extractor='NumWordsRulesExtractor', html=data)
        text = extractor.getText()
    return str(text).lower()

with open(metadata_directory) as f:
    metadata = f.readlines()
en_hash_values = {}
for i in metadata:
    i = json.loads(i)
    if i['language'] == 'en':
        en_hash_values[i['hash']] = (i['url'], i['path'])

df = pd.read_csv('labels.txt')

filenames = df.file.values
urls = []
for i in df.file.values:
    i = i.split('.')[0]
    if i in en_hash_values:
        urls.append(en_hash_values[i][0])
    else:
        urls.append(' ')
df['url'] = urls

text = []
number_of_segments = []
lengths = []
tokenizer = RegexpTokenizer('[A-Z][a-z0-9]+|[A-Z0-9]+|[a-z0-9]+')
for row in df.itertuples():
    url = row.url
    url = unquote(url)
    url = url.split('/')[3:]
    number_of_segments.append(len(url))
    url = ' '.join(url).strip()
    lengths.append(len(url))
    url = tokenizer.tokenize(url.lower())
    text.append(url)

df['lengths'] = lengths
df['text'] = text
df['number_of_segments'] = number_of_segments

english_words = set(words.words())
#
vocab = {}
counter = 0
for i in text:
    for word in i:
        if word in english_words and word not in vocab:
            vocab[word] = counter
            counter += 1

vocab_length = len(vocab)
doc_length = len(text)
print('url vocab '+str(vocab_length))
print('text '+str(doc_length))

with open('url_vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

tf = np.zeros((doc_length, vocab_length+3))
for i, row in enumerate(text):
    for j, word in enumerate(row):
        if word in vocab:
            tf[i, vocab[word]] += 1

idfs = np.count_nonzero(tf, axis=0)
for i in range(0, vocab_length):
    tf[:,i] = tf[:,i]*math.log(doc_length/(1+idfs[i]))
tf[:,-3] = df['lengths']
tf[:,-2] = df['number_of_segments']

documents = []

def remove_non_ascii_punctuation_stopwords(words):
    new_words = []
    for word in words:
        # Remove non-ASCII characters from list of tokenized words
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        # Remove punctuation from list of tokenized words
        new_word = re.sub(r'[^\w\s]', '', new_word)
        new_word = new_word.strip()
        if new_word == '':
            continue
        # Remove Stopwords
        if new_word in set(stopwords.words('english')):
            continue
        new_words.append(new_word)
    return new_words

for filename in tqdm(filenames):
    text = get_text(directory+en_hash_values[filename.split('.')[0]][1]+'/'+filename)
    hash_val = filename.split('.')[0]
    if hash_val not in en_hash_values:
        documents.append([])
        continue
    words = nltk.word_tokenize(text)
    words = remove_non_ascii_punctuation_stopwords(words)
    documents.append(words)

vocab = {}
counter = 0
for i in documents:
    for word in i:
        if word in english_words and word not in vocab:
            vocab[word] = counter
            counter += 1

vocab_length = len(vocab)
doc_length = len(documents)

print('doc vocab '+str(vocab_length))
print('doc length '+str(doc_length))


with open('doc_vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

tf_doc = np.zeros((doc_length, vocab_length))
for i, row in enumerate(documents):
    for j, word in enumerate(row):
        if word in vocab:
            tf_doc[i, vocab[word]] += 1

if np.isnan(tf_doc).any():
    print('tf_doc contains nan')


idfs_doc = np.count_nonzero(tf_doc, axis=0)

for i in range(0, vocab_length):
    tf_doc[:,i] = tf_doc[:,i]*math.log(doc_length/(1+idfs_doc[i]))

if np.isnan(tf_doc).any():
    print('tf_doc after idf calculation contains nan')

tf_full = np.hstack((tf_doc, tf))
#tf_full = tf

for i in range(0, len(tf_full[0])-1):
    tf_full[:,i] = (tf_full[:,i] - np.mean(tf_full[:,i]))/np.std(tf_full[:,i])


if np.isnan(tf_full).any():
    print('tf_full contains nan')

print(tf_full)
with open('tf_full.pkl', 'wb') as f:
    pickle.dump(tf_full, f)


tf_full[:,-1] = df.label.values
prec_0 = 0
rec_0 = 0
prec_1 = 0
rec_1 = 0
sup_0 = 0
sup_1 = 0
f1_0 = 0
f1_1 = 0
for i in range(0, 5):
    randomize = np.arange(len(tf_full))
    np.random.shuffle(randomize)
    tf_full = tf_full[randomize]
    filenames = filenames[randomize]

    train_x = tf_full[0:960,0:-1]
    train_y = tf_full[0:960:,-1]
    test_x = tf_full[960:1200, 0:-1]
    test_y = tf_full[960:1200, -1]
    dev_x = tf_full[1200:, 0:-1]
    dev_y = tf_full[1200:, -1]
    dev_filenames = filenames[1200:]
    sm = SMOTE(random_state=42)
    x_res, y_res = sm.fit_resample(train_x, train_y)

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, min_samples_split=2).fit(train_x, train_y)
    with open('rforest.pkl','wb') as f:
        pickle.dump(clf,f)
    preds = clf.predict(dev_x)

    rep = classification_report(dev_y, preds, output_dict=True)
    prec_0 = prec_0 + rep['0.0']['precision']
    prec_1 = prec_1 + rep['1.0']['precision']
    rec_0 = rec_0 + rep['0.0']['recall']
    rec_1 = rec_1 + rep['1.0']['recall']
    f1_0 = f1_0 + rep['0.0']['f1-score']
    f1_1 = f1_1 + rep['1.0']['f1-score']
    sup_0 = sup_0 + rep['0.0']['support']
    sup_1 = sup_1 + rep['1.0']['support']


print('precision 0 '+ str(prec_0/5))
print('recall 0 '+ str(rec_0/5))
print('precision 1 '+str(prec_1/5))
print('recall 1 '+ str(rec_1/5))
print('f1 0 '+str(f1_0/5))
print('f1 1 '+str(f1_1/5))
print('support 0 '+str(sup_0/5))
print('support 1 '+str(sup_1/5))