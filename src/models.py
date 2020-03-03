import itertools
import logging
import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
from sklearn.externals import joblib
import pandas as pd
 # Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import os
from pathlib import Path
# from src.distances import get_most_similar_documents

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


PATH_DICTIONARY = "models/id2word.dictionary"
PATH_CORPUS = "models/corpus.mm"
PATH_LDA_MODEL = "models/LDA.model"
PATH_DOC_TOPIC_DIST = "models/doc_topic_dist.dat"


def head(stream, n=10):
    """
    Return the first `n` elements of the stream, as plain list.
    """
    return list(itertools.islice(stream, n))


def tokenize(text, STOPWORDS):
    # deacc=True to remove punctuations
    return [token for token in simple_preprocess(text, deacc=True)
            if token not in STOPWORDS]


def make_texts_corpus(sentences):
    for sentence in sentences:
        yield simple_preprocess(sentence, deacc=True)


class StreamCorpus(object):
    def __init__(self, sentences, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` documents
        Yield each document in turn, as a list of tokens.
        """
        self.sentences = sentences
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        for tokens in itertools.islice(make_texts_corpus(self.sentences),
                                       self.clip_docs):
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs


class LDAModel:

    def __init__(self, num_topics,
                 random_state=100,
                 update_every=1, 
                 alpha='auto',
                 per_word_topics=False):
        """
        :param sentences: list or iterable (recommend)
        """

        # data
        self.sentences = None

        # params
        self.lda_model = None
        self.dictionary = None
        self.corpus = None

        # hyperparams
        self.num_topics = num_topics
        self.random_state = random_state
        self.update_every = update_every
        self.alpha = alpha
        self.per_word_topics = per_word_topics

    def make_sentences(self, data):
        print("===make_sentences===")
        for item in data:
            yield item

    def fit(self, data):
        print("====fit===")
        from itertools import tee
        sentences = self.make_sentences(data)
        sentences = make_texts_corpus(sentences)
        self.id2word = gensim.corpora.Dictionary(sentences)
        self.id2word.filter_extremes(no_below=10, no_above=0.25)
        self.id2word.compactify()
        self.id2word.save(PATH_DICTIONARY)
        sentences = self.make_sentences(data)
        cospus = StreamCorpus(sentences, self.id2word)
        gensim.corpora.MmCorpus.serialize('PATH_CORPUS', cospus)
        self.corpus = gensim.corpora.MmCorpus('PATH_CORPUS')
        path = Path.cwd()
        path = os.path.join(path, "mallet-2.0.8\mallet-2.0.8")
        bin__ = os.path.join(path,  'bin\mallet')
        os.environ.update({'MALLET_HOME': r"{}".format(path)})
        mallet_path = r"{}".format(bin__) # update this path
        # self.lda_model = gensim.models.ldamodel.LdaModel(
        #                                    corpus=self.corpus,
        #                                    id2word=self.id2word,
        #                                    num_topics=self.num_topics, 
        #                                    random_state=100,
        #                                    update_every=1,
        #                                    chunksize=len(data),
        #                                    passes=10,
        #                                    alpha='auto',
        #                                    per_word_topics=True)
        self.lda_model = gensim.models.wrappers.LdaMallet(
            mallet_path, 
            corpus=self.corpus, 
            num_topics=self.num_topics, 
            id2word=self.id2word)
        print("===done training===")
        self.lda_model.save(PATH_LDA_MODEL)

        sentences = self.make_sentences(data)
        sentences = make_texts_corpus(sentences)
        df_topic_sents_keywords = self.format_topics_sentences(ldamodel=self.lda_model, corpus=self.corpus, texts=sentences)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        # load csv
        df = pd.read_csv('data.csv')
        df['cluster'] = df_dominant_topic.Dominant_Topic
        df['score'] = df_dominant_topic.Topic_Perc_Contrib
        df['keywords'] = df_dominant_topic.Keywords
        df.to_csv('result.csv')

        self.picture()

    def picture(self):
        print("===picture===")
        # Visualize the topics
        # pyLDAvis.enable_notebook()
        model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(self.lda_model)
        vis = pyLDAvis.gensim.prepare(model, self.corpus, self.id2word)
        pyLDAvis.save_html(vis, 'lda.html')


if __name__ == '__main__':
    main()
