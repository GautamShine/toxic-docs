#!/usr/bin/env python

"""
__author__ = 'Gautam Shine'
__email__ = 'gshine@stanford.edu'

Document classifier for the "Toxic Docs" repository from Columbia University
and the Center for Public Integrity. Data set consists of PDF files of
emails, memos, advertisements, news articles, scientific articles cited in
legal cases involving allegations of environmental harm from toxic substances.

"""

import bson
import time
import re
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt, rcParams
from spacy.en import English

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics

"""
NLP functions (NER, lemmatization) for obtaining clean tokens
"""
class NLP():

    def __init__(self):
        self.spacy = English()

    """
    Normalizes emails, people, places, and organizations
    """
    def preprocessor(self, token):

        # normalize emails
        if token.like_email:
            return 'local@domain'

        # normalize names
        elif token.ent_type == 346:
            return 'ne_person'

        # normalize places
        elif token.ent_type == 350 or token.ent_type == 351:
            return 'ne_place'

        # normalize organizations
        elif token.ent_type == 349:
            return 'ne_org'

        return token.lemma_

    """
    Tokenizes input string with preprocessing
    """
    def tokenizer(self, doc):
        try:
            spacy_doc = self.spacy(doc[:300])
            return [self.preprocessor(t) for t in spacy_doc \
                    if (t.is_alpha or t.like_email) \
                    and len(t) < 50 and len(t) > 2 \
                    and not (t.is_punct or t.is_stop)]
        except:
            print(doc)
            raise('Error: failed to tokenize a document')


"""
Data transformation functions to go from database dump to text features
"""
class DataProcessor():

    def __init__(self):
        self.nlp = NLP()
        self.label_dict = {'email' : 0, 'internal_memo': 1,
                'boardroom_minutes': 2, 'annual_report': 3,
                'public_relations': 4, 'general_correspondance': 5,
                'newspaper_article': 6,'deposition': 7,
                'scientific_article_unpublished': 8,
                'scientific_article_published': 9,
                'advertisement': 10, 'trade_association': 11}
        self.inv_label_dict = {v: k for k, v in self.label_dict.items()}

    """
    Takes a bson and returns the corresponding list of dicts, the frequency of
    each label, and the number of unlabeled items
    """
    def load_bson(self, bson_file, label_key):

        # 'rb' for read as binary
        f = open(bson_file, 'rb')
        docs = bson.decode_all(f.read())

        labels = np.zeros((len(docs), 1))
        counts = defaultdict(int)

        for i in range(len(docs)):
            try:
                labels[i] = self.label_dict[docs[i][label_key]]
                counts[docs[i][label_key]] += 1
            except:
                labels[i] = -1
                counts['unlabeled'] += 1

        return docs, labels, counts

    """
    Applies a TF-IDF transformer and count vectorizer to the corpus to build
    n-gram features for classification
    """
    def vectorize(self, docs, text_key, min_df=2, max_ngram=2):

        docs = [x[text_key] for x in docs]
        vectorizer = TfidfVectorizer(min_df=min_df, \
                ngram_range=(1, max_ngram), tokenizer=self.nlp.tokenizer)

        return vectorizer, vectorizer.fit_transform(docs), vectorizer.get_feature_names()

    """
    Splits the labeled subset into train and test sets
    """
    def split_data(self, y_all, X_all, split=0.7, seed=0):
        
        y_valid = y_all[y_all != -1]
        X_valid = X_all[(y_all != -1).flatten()]

        X_train, X_test, y_train, y_test = \
                train_test_split(X_valid, y_valid, train_size=split, random_state=seed)

        return y_train, X_train, y_test, X_test


class ModelEvaluator():

    def __init__(self):
        self.performance = {}

    """
    Trains a given classifier
    """
    def train(self, model, y_train, X_train):

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        train_score = model.score(X_train, y_train)

        return train_score, train_time

    """
    Tests a given fitted classifier
    """
    def test(self, model, y_test, X_test):

        t0 = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - t0
        test_accuracy = metrics.accuracy_score(y_test, y_pred)
        test_precision = metrics.precision_score(y_test, y_pred, average=None)
        test_recall = metrics.recall_score(y_test, y_pred, average=None)

        return test_accuracy, test_precision, test_recall, test_time

    """
    Utility function for printing class-wise precision/recall
    """
    def print_scores(self, dp, acc, prec, rec):

        print('Accuracy:', acc)
        for i in range(len(prec)):
            print('{0} \n F1: {1: .3f}, P: {2: .3f}, R: {3: .3f}, '.format( \
                    dp.inv_label_dict[i], 2/(1/prec[i] + 1/rec[i]), prec[i], rec[i]))
        print('\n')


"""
Propagates labels to unlabeled data points for semisupervised learning
"""
class Propagator():

    """
    Scalarizes the vector of distances from hyperplane to a confidence measure
    """
    def confidence_score(self, hp_dists):

        # TODO: vectorize this
        ranking = np.argsort(hp_dists)
        return hp_dists[ranking[0]] - hp_dists[ranking[1]]

    """
    Assigns labels where confidence exceeds a given threshold
    """
    def propagate_labels(self, X_unlabeled, conf_scores, conf_threshold=1.5):

        # TODO: implement
        y_labeled = -np.ones((X_unlabeled.shape[0],1))
        return y_labeled


"""
Main
"""
if __name__ == '__main__':

    bson_file = 'documents.bson'
    label_key = 'document_type'
    text_key = 'text'

    # Process the raw data
    dp = DataProcessor()
    docs, y_all, counts = dp.load_bson(bson_file, label_key)
    t0 = time.time()
    vectorizer, X_all, feat_names = dp.vectorize(docs, text_key, min_df=2, max_ngram=2)
    vec_time = time.time() - t0

    y_train, X_train, y_test, X_test = dp.split_data(y_all, X_all, split=0.7, seed=0)
    me = ModelEvaluator()

    print('Vectorization time:', vec_time)
    print('Data matrix size:', X_all.shape)
    print(dp.label_dict, '\n')

    # Multinomial Naive Bayes
    MNB = MultinomialNB()
    MNB_train_acc, MNB_train_time = me.train(MNB, y_train, X_train)
    MNB_test_acc, MNB_test_prec, MNB_test_rec, MNB_test_time = me.test(MNB, y_test, X_test)
    print('MNB time:', MNB_train_time)
    me.print_scores(dp, MNB_test_acc, MNB_test_prec, MNB_test_rec)

    # Bernoulli Naive Bayes
    BNB = BernoulliNB()
    BNB_train_acc, BNB_train_time = me.train(BNB, y_train, X_train)
    BNB_test_acc, BNB_test_prec, BNB_test_rec, BNB_test_time = me.test(BNB, y_test, X_test)
    print('BNB time:', BNB_train_time)
    me.print_scores(dp, BNB_test_acc, BNB_test_prec, BNB_test_rec)

    # LinearSVC (liblinear SVM implementation, one-v-all)
    # TODO: test hyperparameters
    SVM = LinearSVC()
    SVM_train_acc, SVM_train_time = me.train(SVM, y_train, X_train)
    SVM_test_acc, SVM_test_prec, SVM_test_rec, SVM_test_time = me.test(SVM, y_test, X_test)
    print('SVM time:', SVM_train_time)
    me.print_scores(dp, SVM_test_acc, SVM_test_prec, SVM_test_rec)
    hp_dists = SVM.decision_function(X_train)

    # SVM - linear, class-weighted
    SVMcw = SVC(kernel='linear', class_weight='balanced')
    SVMcw_train_acc, SVMcw_train_time = me.train(SVMcw, y_train, X_train)
    SVMcw_test_acc, SVMcw_test_prec, SVMcw_test_rec, SVMcw_test_time = me.test(SVMcw, y_test, X_test)
    print('SVMcw time:', SVMcw_train_time)
    me.print_scores(dp, SVMcw_test_acc, SVMcw_test_prec, SVMcw_test_rec)

    x = np.arange(len(counts))
    y = counts.values()
    ymax = max(y)*1.1

    plt.bar(x, y, align='center', width=0.5)
    plt.ylim(0, ymax)
    plt.xticks(x, counts.keys(), rotation=45, ha='right')
    rcParams.update({'figure.autolayout': True, 'font.size': 30})

    #plt.show()
