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
from src.crawl import Crawler
from pathlib import Path
from threading import Thread
import threading
import time
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import timeit

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

def crawl_data(arr, links):
    # q = Queue()
    # for index,link in enumerate(links):
        # print("===crawl===", index)
        # try:
            # t = threading.Thread(target=onCraw, args = (q,link))
            # t.daemon = True
            # t.start()
        # except:
        #     print ("error")
        # columns = {}
        # try:
        #     response = requests.get(link, verify=True, timeout=1)
        #     paragraphs = justext.justext(response.content, justext.get_stoplist("Vietnamese"))
        #     content = ''
        #     for paragraph in paragraphs:
        #         if not paragraph.is_boilerplate:
        #             content = content + ' ' + paragraph.text
        #     columns['link'] = link
        #     columns['content'] = raw_to_text(content)
        #     data.append(columns)
        #     print(index,end =" ")
        # except:
        #     print("An exception occurred", index, end =" ")
        # try:
        # results = pool.map(partial(onCraw, data = data), links)
    # data.append(q.get())
    # print(q.get())
    return arr
def onCraw(link):
    print("===crawl===")
    print(link)
    columns = {}
    # print(link)
    try:
        response = requests.get(link, verify=True, timeout=1)
        paragraphs = justext.justext(response.content, justext.get_stoplist("Vietnamese"))
        content = ''
        for paragraph in paragraphs:
            if not paragraph.is_boilerplate:
                content = content + ' ' + paragraph.text
                columns['link'] = link
                columns['content'] = raw_to_text(content)
        # print(content)
        # q.put(columns)
        arr.append(columns)
    # print(index,end =" ")
    except:
        print("An exception occurred", end =" ")
    return 0

def create_csv(data, file_name):
    df = pd.DataFrame(data)
    df = df.dropna()
    print(df.head())
    df.to_csv(file_name)
    return file_name
    
def visualize():
    return

def read_file(file_name):
    f = open(file_name, 'r', encoding='utf8', errors='ignore') 
    file_lines = f.readlines()
    f.close()
    # links = [item for item in file_lines if '.htm' in item] 
    # links = [item for item in links if '.aspx' in item] 
    links = ['http://' + item.replace(' ','').replace('\n','').replace('"','') for item in file_lines if 'http' not in item]
    return links

def main():
    links = read_file(filename)
    arr = []
    # pool.map(onCraw, links)
    start = timeit.default_timer()
    crawler = Crawler(arr, links[:10000])
    crawler.prepareCrawl()
    # data = crawl_data(arr, links[:2000])
    # print(len(crawler.getResult()))
    create_csv(crawler.getResult(), 'dataa.csv')
    df = pd.read_csv('dataa.csv')
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    # df = pd.read_csv('dantri.csv')
    df = df.dropna()
    print(df.shape)

    # data = df.content.values.tolist()
    # model = LDAModel(int(num_topics))
    # model.fit(data)
    # print(Path.cwd())

  

if __name__ == "__main__":
    filename = sys.argv[1]
    num_topics = sys.argv[2]
    main()
   
