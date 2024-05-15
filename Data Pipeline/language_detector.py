import langid
from tqdm import tqdm
from bs4 import BeautifulSoup
from bs4.element import Comment
import os
import json

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
        try:
            soup = BeautifulSoup(data, 'html.parser')
            text = soup.findAll(text=True)
        except Exception as e:
            return ''
        visible_texts = filter(tag_visible, text)
        text = " ".join(t.strip() for t in visible_texts)
    return text

def get_urls():
    success_urls = {}
    with open('crawled_urls') as f:
        success = f.readlines()
    for line in success:
        line = json.loads(line)
        if 'hash' in line:
            success_urls[line['hash']] = (line['response'], line['folder_number'])
    return success_urls

directory = ''
success_urls = get_urls()

with open('metadata', 'w', encoding='utf-8') as f:
    for hash_val in tqdm(success_urls):
        text = get_text(directory+success_urls[hash_val][1]+'/'+hash_val+'.html')
        try:
            detector = langid.classify(text)
            if len(text) > 10:
                language = detector[0]
            else:
                language = 'un'
            confidence = detector[1]
            f.write(json.dumps({'hash': hash_val, 'language':language, 'confidence':confidence})+'\n')
        except Exception as e:
            continue
