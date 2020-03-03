from multiprocessing.dummy import Pool as ThreadPool
import requests
import justext
import re
from pyvi import ViTokenizer

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
    return re.sub(  "\s\s+", " ", text)


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

class Crawler:
    def __init__(self,results, links):

        self.results = results
        self.links = links
        self.count = 0

    def onCraw(self, link):
        columns = {}
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
            self.results.append(columns)
            self.count = self.count + 1
            print(self.count, end = " ")
        # print(index,end =" ")
        except:
            print("An exception occurred", end =" ")
        return 0

    def prepareCrawl(self):
        pool = ThreadPool(1000)
        pool.map(self.onCraw, self.links)

    def getResult(self):
        return self.results