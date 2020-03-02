import sys
import requests
import justext
import pandas as pd
import logging
import re
from bs4 import BeautifulSoup
from markdown import markdown
from pyvi import ViTokenizer
import itertools
import gensim
from gensim.utils import simple_preprocess
from src.models import LDAModel
from pathlib import Path

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


def raw_to_text(text, parser="html.parser",
                     tags=['pre', 'code', 'a', 'img', 'i']):
    
    text = remove_links_content(text)
    text = remove_emails(text)
    text = remove_punctuation(text)
    text = text.replace('\n', ' ')
    text = text.replace('"', '')
    text = text.replace('”','')
    text = text.replace('“','')
    text = text.replace('’','')
    text = text.replace('‘','')
    text = remove_numeric(text)
    text = remove_multiple_space(text)
    text = text.lower().strip()
    text = ViTokenizer.tokenize(text)
    text = remove_stopwords(text, stopwords=import_stopwords())
    return text

def import_stopwords():
    file = open('data/vni_stopwords.txt', encoding="utf8")
    stopwords=[]
    for line in file:
        stopwords.append("_".join(line.strip().split()))
    return stopwords


def remove_emails(text):
    return re.sub('\S*@\S*\s?', '', text)


def remove_links_content(text):
    text = re.sub(r"http\S+", "", text)
    return text


def remove_multiple_space(text):
    return re.sub("\s\s+", " ", text)


def remove_punctuation(text):
    """https://stackoverflow.com/a/37221663"""
    import string  # noqa
    table = str.maketrans({key: None for key in string.punctuation})
    return text.translate(table)


def remove_numeric(text):
    import string  # noqa
    table = str.maketrans({key: None for key in string.digits})
    return text.translate(table)


def remove_stopwords(text, stopwords):
    return " ".join([word for word in text.split() if word not in stopwords])


def crawl_data(links):
    print(links[:10])
    data = []
    for index,link in enumerate(links):
        columns = {}
        try:
            print("===crawl===", index)
            response = requests.get(link)
            paragraphs = justext.justext(response.content, justext.get_stoplist("Vietnamese"))
            content = ''
            for paragraph in paragraphs:
                if not paragraph.is_boilerplate:
                    content = content + ' ' + paragraph.text
            columns['link'] = link
            columns['content'] = raw_to_text(content)
            data.append(columns)
        except:
            print("An exception occurred", index)
    return data

def create_csv(data, file_name):
    df = pd.DataFrame(data)
    df = df.dropna()
    df.to_csv(file_name)
    return file_name
    
def visualize():
    return

def read_file(file_name):
    f = open(file_name, 'r', encoding='utf8', errors='ignore') 
    file_lines = f.readlines()
    f.close()
    links = [item for item in file_lines if '.htm' in item] 
    links = [item for item in links if '.aspx' in item] 
    links = ['http://' + item.replace(' ','').replace('\n','') for item in links if 'http' not in item]
    return links

def main():
    # links = read_file(filename)
    # data = crawl_data(links[:5])
    # create_csv(data, 'data.csv')
    df = pd.read_csv('data.csv')
    df = pd.read_csv('dantri.csv')
    df = df.dropna()
    print(df.isna().sum())
    data = df.content.values.tolist()
    model = LDAModel(num_topics)
    model.fit(data)
    print(Path.cwd())

  

if __name__ == "__main__":
    filename = sys.argv[1]
    num_topics = sys.argv[2]
    main()
   
