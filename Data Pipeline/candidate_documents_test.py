import json
from urllib.parse import unquote
from nltk.tokenize import RegexpTokenizer
import numpy as np
import math
import nltk
import re, unicodedata
from nltk.corpus import stopwords
from joblib import load
import pickle
import multiprocessing
from tqdm import tqdm
import os

with open('url_vocab.pkl', 'rb') as f:
    url_vocab = pickle.load(f)
with open('doc_vocab.pkl', 'rb') as f:
    doc_vocab = pickle.load(f)
url_vocab_length = len(url_vocab)
doc_vocab_length = len(doc_vocab)

BATCH_SIZE = 1000
FEATURE_SIZE = url_vocab_length+doc_vocab_length+2
DOC_FEATURE_START = url_vocab_length+2
DIRECTORY = '/data/mus824/data/third_crawl/'
tokenizer = RegexpTokenizer('[A-Z][a-z0-9]+|[A-Z0-9]+|[a-z0-9]+')

en_hash_values = {}
with open(DIRECTORY+'language_metadata', encoding='utf-8') as f:
    metadata = f.readlines()
for line in metadata:
    line = json.loads(line)
    if line['language'] == 'en':
        en_hash_values[line['hash']] = ''


with open(DIRECTORY+'success', encoding='utf-8') as f:
    metadata = f.readlines()
for i in metadata:
    i = json.loads(i)
    if i['hash'] in en_hash_values:
        en_hash_values[i['hash']] = i['response']

print("Number of candidate privacy policies: "+str(len(en_hash_values)))
filenames = list(en_hash_values.keys())
total_number_of_documents = len(filenames)
#def batch_generator(batch_size, filenames):
#    for i in range(0, total_number_of_documents, batch_size):
#        yield filenames[i:i+batch_size]

def url_processor(url):
    # print('url_processor')
    try:
        url = unquote(url)
        url = url.split('/')[3:]
        segment_length = len(url)
        url = ' '.join(url).strip()
        url_length = len(url)
        url = tokenizer.tokenize(url.lower())
        return url, segment_length, url_length
    except:
        print('No url!')
        return [], 0, 0

def remove_non_ascii_punctuation_stopwords(words):
    # print('remove')
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


def process_docs(line):
    line = json.loads(line)
    text = line['text']
    try:
        #text = get_text(DIRECTORY+en_hash_values[filename]['directory']+'/'+filename+'.html')
        words = nltk.word_tokenize(text)
        words = remove_non_ascii_punctuation_stopwords(words)
        return words
    except Exception as e:
        print(e)
        return []

def term_frequency_counter(row):
    # print('term_frequency_counter')
    frequencies = np.zeros((url_vocab_length))
    for word in row:
        if word in url_vocab:
            frequencies[url_vocab[word]] += 1
    return frequencies


def term_frequency_counter_docs(row):
    # print('term_frequency_counter_docs')
    frequencies = np.zeros((doc_vocab_length))
    for word in row:
        if word in doc_vocab:
            frequencies[doc_vocab[word]] += 1
    return frequencies

p = multiprocessing.Pool(16)
#batches_generated = batch_generator(BATCH_SIZE, filenames)

TEXT_DIRECTORY = 'boilerpipe-policy-text/'
batches_generated = os.listdir(DIRECTORY+TEXT_DIRECTORY)
for batched_files in tqdm(batches_generated):
    count = 0
    with open(DIRECTORY+TEXT_DIRECTORY+batched_files, encoding='utf-8') as f:
        data = f.readlines()
    processed_urls = []
    for line in data:
        line = json.loads(line)
        count += 1
        processed_urls.append(en_hash_values[line['hash']])

    number_of_segments = []
    url_lengths = []

    multithread_iterable = p.imap(url_processor, processed_urls)
    for i, return_value in enumerate(multithread_iterable):
        processed_urls[i] = return_value[0]
        number_of_segments.append(return_value[1])
        url_lengths.append(return_value[2])

    tf = np.zeros((count, FEATURE_SIZE))

    multithread_iterable = p.imap(term_frequency_counter, processed_urls)
    for i, return_value in enumerate(multithread_iterable):
        tf[i, 0:url_vocab_length] = return_value

    idfs = np.count_nonzero(tf[:,0:url_vocab_length], axis=0)
    tf[:,url_vocab_length] = url_lengths
    tf[:,url_vocab_length+1] = number_of_segments
    for i in range(0, url_vocab_length):
        tf[:,i] = tf[:,i]*math.log(count/(1+idfs[i]))

    documents = []
    multithread_iterable = p.imap(process_docs, data)
    for i, return_value in enumerate(multithread_iterable):
        documents.append(return_value)

    multithread_iterable = p.imap(term_frequency_counter_docs, documents)
    for i, return_value in enumerate(multithread_iterable):
        tf[i, DOC_FEATURE_START:FEATURE_SIZE] = return_value

    idfs_doc = np.count_nonzero(tf[:, DOC_FEATURE_START:FEATURE_SIZE], axis=0)

    for i in range(DOC_FEATURE_START, FEATURE_SIZE):
        tf[:,i] = tf[:,i]*math.log(count/(1+idfs_doc[i-DOC_FEATURE_START]))

    for i in range(0, FEATURE_SIZE):
        if np.std(tf[:,i]) != 0:
            tf[:,i] = (tf[:,i] - np.mean(tf[:,i]))/np.std(tf[:,i])

    rfclassifier = load('rforest.pkl')
    probabilities = rfclassifier.predict_proba(tf)
    print(len(probabilities))
    print(len(data))
    print(count)
    with open('probabilities', 'a+', encoding='utf-8') as f:
        for i in range(0, count):
            f.write(json.loads(data[i])['hash']+' '+str(probabilities[i][1])+'\n')

