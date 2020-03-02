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

        # init model
        # self._make_dictionary()
        # self._make_corpus_bow()

    def make_sentences(self, data):
        print("===make_sentences===")
        for item in data:
            yield item

    def _make_corpus_bow(self, sentences):
        print("===_make_corpus_bow===")
        cospus = StreamCorpus(sentences, self.id2word)
        # save corpus
        gensim.corpora.MmCorpus.serialize(PATH_CORPUS, cospus)
        self.corpus = gensim.corpora.MmCorpus(PATH_CORPUS)

    def _make_corpus_tfidf(self):
        pass

    def _make_dictionary(self, sentences):
        print("===_make_dictionary===")
        self.texts_corpus = make_texts_corpus(sentences)
        self.id2word = gensim.corpora.Dictionary(self.texts_corpus)
        self.id2word.filter_extremes(no_below=10, no_above=0.25)
        self.id2word.compactify()
        self.id2word.save(PATH_DICTIONARY)

    def documents_topic_distribution(self):
        doc_topic_dist = np.array(
            [[tup[1] for tup in lst] for lst in self.lda_model[self.corpus]]
        )
        # save documents-topics matrix
        joblib.dump(doc_topic_dist, PATH_DOC_TOPIC_DIST)
        return doc_topic_dist

    def fit(self, data):
        print("====fit===")
        from itertools import tee
        # sentences_1, sentences_2 = tee(sentences)
        sentences = self.make_sentences(data)
        sentences = make_texts_corpus(sentences)
        self.id2word = gensim.corpora.Dictionary(sentences)
        self.id2word.filter_extremes(no_below=10, no_above=0.25)
        self.id2word.compactify()
        self.id2word.save(PATH_DICTIONARY)
        # self._make_dictionary(sentences)
        sentences = self.make_sentences(data)
        # self._make_corpus_bow(sentences)
        cospus = StreamCorpus(sentences, self.id2word)
        gensim.corpora.MmCorpus.serialize('PATH_CORPUS', cospus)
        self.corpus = gensim.corpora.MmCorpus('PATH_CORPUS')
        # print(self.corpus)
        # print(self.id2word)
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
            num_topics=3, 
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
        df = pd.read_csv('dantri.csv')
        df['cluster'] = df_dominant_topic.Dominant_Topic
        df['score'] = df_dominant_topic.Topic_Perc_Contrib
        df['keywords'] = df_dominant_topic.Keywords
        df.to_csv('dantri.csv')

        self.picture()

    def picture(self):
        print("===picture===")
        # Visualize the topics
        # pyLDAvis.enable_notebook()
        model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(self.lda_model)
        vis = pyLDAvis.gensim.prepare(model, self.corpus, self.id2word)
        pyLDAvis.save_html(vis, 'lda.html')

    def transform(self, sentence):
        """
        :param document: preprocessed document
        """
        document_corpus = next(make_texts_corpus([sentence]))
        corpus = self.id2word.doc2bow(document_corpus)
        document_dist = np.array(
            [tup[1] for tup in self.lda_model.get_document_topics(bow=corpus)]
        )
        return corpus, document_dist

    # def predict(self, document_dist):
    #     doc_topic_dist = self.documents_topic_distribution()
    #     return get_most_similar_documents(document_dist, doc_topic_dist)

    def update(self, new_corpus):  # TODO
        """
        Online Learning LDA
        https://radimrehurek.com/gensim/models/ldamodel.html#usage-examples
        https://radimrehurek.com/gensim/wiki.html#latent-dirichlet-allocation
        """
        self.lda_model.update(new_corpus)
        # get topic probability distribution for documents
        for corpus in new_corpus:
            yield self.lda_model[corpus]

    # def model_perplexity(self):
    #     logging.INFO(self.lda_model.log_perplexity(self.corpus))

    # def coherence_score(self):
    #     self.coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(
    #         model=self.lda_model, texts=self.corpus,
    #         dictionary=self.id2word, coherence='c_v'
    #     )
    #     logging.INFO(self.coherence_model_lda.get_coherence())

    def compute_coherence_values(self, mallet_path, dictionary, corpus,
                                 texts, end=40, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        end : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model
                           with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, end, step):
            model = gensim.models.wrappers.LdaMallet(
                mallet_path, corpus=self.corpus,
                num_topics=self.num_topics, id2word=self.id2word)
            model_list.append(model)
            coherencemodel = gensim.models.coherencemodel.CoherenceModel(
                model=model, texts=self.texts_corpus,
                dictionary=self.dictionary, coherence='c_v'
            )
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def plot(self, coherence_values, end=40, start=2, step=3):
        x = range(start, end, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

    def print_topics(self):
        pass

    def format_topics_sentences(self, ldamodel, corpus, texts):
        print("===format_topics_sentences===")
        # Init output
        sent_topics_df = pd.DataFrame()
        # print(next(self.lda_model[self.corpus]))
        # Get main topic in each document
        for i, row in enumerate(self.lda_model[self.corpus]):
            print("row", row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        print(sent_topics_df[:10])
        print("assign columns===format_topics_sentences===")
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        print("Done===format_topics_sentences===")
        return(sent_topics_df)


def main():
    # TODO
    sentences = None
    sentences = make_texts_corpus(sentences)
    id2word = gensim.corpora.Dictionary(sentences)
    id2word.filter_extremes(no_below=20, no_above=0.1)
    id2word.compactify()

    # save dictionary
    # id2word.save('path_to_save_file.dictionary')
    cospus = StreamCorpus(sentences, id2word)
    # save corpus
    # gensim.corpora.MmCorpus.serialize('path_to_save_file.mm', cospus)
    # load corpus
    # mm_corpus = gensim.corpora.MmCorpus('path_to_save_file.mm')
    lda_model = gensim.models.ldamodel.LdaModel(
        cospus, num_topics=64, id2word=id2word, passes=10, chunksize=100
    )
    # save model
    # lda_model.save('path_to_save_model.model')
    lda_model.print_topics(-1)


if __name__ == '__main__':
    main()
