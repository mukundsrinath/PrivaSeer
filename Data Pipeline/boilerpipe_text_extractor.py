''''
Extracts text from HTML files using boilerpipe
'''

from bs4 import BeautifulSoup
#from bs4.element import Comment
import json
from tqdm import tqdm
import multiprocessing
from boilerpipe.extract import Extractor
import os

BATCH_SIZE = 20000
DIRECTORY = ''
METATDATA_DIRECTORY = ''

def get_text(i):
    text = ''
    with open(i, encoding='utf-8') as f:
        data = f.read()
    soup = BeautifulSoup(data, 'html.parser')
    if soup.title != None:
        title = soup.title.string
    else:
        title = " "
    if title == None:
        title = ''
    extractor = Extractor(extractor='NumWordsRulesExtractor', html=data)
    text = extractor.getText()
    if text == None:
        text = ''
    return str(title)+"~|~"+str(text)

def process_docs(filename):
    try:
        text = get_text(DIRECTORY+'urls/'+metadata[filename][0]+'/'+filename+'.html')
    except Exception as e:
        text = '~|~'
        print(e)
    return text

metadata = {}
with open(METATDATA_DIRECTORY) as f:
    data = f.readlines()
for line in data:
    line = json.loads(line)
    metadata[line['hash']] = line

filenames = list(metadata.keys())
total_number_of_documents = len(filenames)
def batch_generator(batch_size, filenames):
    for i in range(0, total_number_of_documents, batch_size):
        yield filenames[i:i+batch_size]

batches_generated = batch_generator(BATCH_SIZE, filenames)
counter = 1

for batched_files in tqdm(batches_generated):
    with open(DIRECTORY+'boilerpipe-policy-text/'+str(counter), 'w', encoding='utf-8') as f:
        for _file in batched_files:
            text = process_docs(_file)
            title_p, text_p = text.split('~|~',1)
            data = {'hash':_file, 'text':text_p, 'title':title_p}
            f.write(json.dumps(data) + '\n')
    counter += 1

