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
from scipy import sparse as sp
from matplotlib import pyplot as plt, rcParams
from spacy.en import English
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

"""
NLP functions (NER, lemmatization) for obtaining clean tokens
"""
class NLP():

    def __init__(self, num_chars):
        self.spacy = English()
        self.num_chars = num_chars

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
            if len(doc) > 2*self.num_chars:
                spacy_doc = self.spacy(doc[:self.num_chars] + doc[-self.num_chars:])
            else:
                spacy_doc = self.spacy(doc)
                
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

    def __init__(self, num_chars):
        self.nlp = NLP(num_chars)
        self.label_dict = {'email': 0, 'internal_memo': 1,
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
    def load_bson(self, bson_file, label_key, plot_hist=False):

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

        if(plot_hist):

            x = np.arange(len(counts))
            y = counts.values()
            ymax = max(y)*1.1

            plt.figure()
            plt.bar(x, y, align='center', width=0.5)
            plt.ylim(0, ymax)
            plt.xticks(x, counts.keys(), rotation=45, ha='right')
            rcParams.update({'figure.autolayout': True, 'font.size': 30})

            plt.show()

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
        
        X_unlab = X_all[(y_all == -1).flatten()]

        y_valid = y_all[y_all != -1]
        X_valid = X_all[(y_all != -1).flatten()]

        X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid,\
                train_size=split, random_state=seed)

        return y_train, X_train, y_test, X_test, X_unlab


"""
Utility functions for training and testing sklearn models
"""
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
    Performs a hyperparameter search with cross validation
    """
    def param_search(self, model, param_grid, y_train, X_train, num_folds=5):

        t0 = time.time()
        grid = GridSearchCV(model, param_grid, cv=num_folds)
        grid.fit(X_train, y_train)
        grid_time = time.time() - t0

        return grid.grid_scores_, grid.best_params_, grid_time

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

        return y_pred, test_accuracy, test_precision, test_recall, test_time

    """
    Utility function for printing class-wise precision/recall
    """
    def print_scores(self, dp, acc, prec, rec):

        scores = np.array([[2/(1/prec[i] + 1/rec[i]), prec[i], rec[i]] \
                for i in range(len(prec))])

        print('Accuracy:', acc)
        print('Mean F1:', np.mean(scores[:,0]))
        for i in range(len(prec)):
            print('{0} \n F1: {1: .3f}, P: {2: .3f}, R: {3: .3f}, '.format( \
                    dp.inv_label_dict[i], scores[i][0], scores[i][1], scores[i][2]))
        print('\n')

        return


"""
Propagates labels to unlabeled data points for semisupervised learning
"""
class SemiSupervisedLearner():

    def __init__(self, model):
        self.model = model

    """
    Scalarizes the vector of distances from hyperplane to a confidence measure
    """
    def confidence_scores(self, hp_dists):

        ranking = np.argsort(-hp_dists, axis=1)
        return hp_dists[np.arange(ranking.shape[0]), ranking[:,0]] -\
                hp_dists[np.arange(ranking.shape[0]), ranking[:,1]]

    """
    Grows training set using high confidence predictions from unlabeled set
    """
    def propagate_labels(self, y_working, X_working, y_pred, X_unlab,\
            conf_scores, conf_thresh=1.5):

        add_set = conf_scores > conf_thresh
        y_working = np.concatenate((y_working, y_pred[add_set]))
        X_working = sp.vstack((X_working, X_unlab[add_set]))
        X_unlab = X_unlab[~add_set]

        return y_working, X_working, X_unlab

    """
    Iterates the model and data to grow the training set using soft labels
    """
    def loop_learning(self, X_unlab, y_train, X_train, num_loops, conf_thresh):

        X_working = X_train.copy()
        y_working = y_train.copy()

        for i in range(num_loops):

            self.model.fit(X_working, y_working)

            hp_dists = self.model.decision_function(X_unlab)
            y_pred = np.argmax(hp_dists, axis=1)
            conf_scores = self.confidence_scores(hp_dists)

            y_working, X_working, X_unlab = self.propagate_labels(\
                    y_working, X_working, y_pred, X_unlab,\
                    conf_scores, conf_thresh=conf_thresh)

            print(y_working.shape, X_unlab.shape[0])

        return y_working


"""
Utility functions for understanding and visualizing the data
"""
class DataAnalyzer():

    def __init__(self):
        self.fig_num = -1

    """
    Show all plots
    """
    def show_plots(self):

        plt.show()

        return

    """
    Plot histogram of classes given a y vector
    """
    def class_hist(self, y, labels):

        counts = np.bincount(y)

        x = np.arange(len(counts))
        ymax = max(counts)*1.1

        self.fig_num += 1
        plt.figure(self.fig_num)
        plt.bar(x, counts, align='center', width=0.5)
        plt.ylim(0, ymax)
        plt.xticks(x, labels, rotation=45, ha='right')
        rcParams.update({'figure.autolayout': True, 'font.size': 25})

        return

    """
    Returns mean confidence of each class 
    """
    def class_confidence(self, y, conf_scores):

        df = pd.DataFrame({'y': y, 'cs': conf_scores})
        class_confs = df.groupby('y').mean()
        class_confs = class_confs.as_matrix()

        return class_confs

    """
    Retrieve the most important features for predicting a given class
    """
    def important_feats(self, model, feat_names, labels, num_feats=5):

        feat_ranking = np.argsort(-model.coef_, axis=1)
        if type(feat_names) is list:
            feat_names = np.array(feat_names)

        for i in range(len(labels)):

            print(labels[i])
            print(feat_names[feat_ranking[i,:]][:num_feats])

        return

    """
    Randomly sample a few documents from a given class labeled correctly or not
    """
    def sample_docs(self, docs, y, y_pred, target_y, pred_correct=False, num_docs=3, seed=0):

        sample_inds = []

        for i in range(num_docs):

            print(docs[sample_inds[i]])

        return


"""
Main
"""
if __name__ == '__main__':

    bson_file = 'documents.bson'
    label_key = 'document_type'
    text_key = 'text'

    # Process the raw data
    dp = DataProcessor(num_chars=300)
    docs, y_all, counts = dp.load_bson(bson_file, label_key)
    t0 = time.time()
    vectorizer, X_all, feat_names = dp.vectorize(docs, text_key, min_df=2, max_ngram=2)
    vec_time = time.time() - t0

    y_train, X_train, y_test, X_test, X_unlab = dp.split_data(y_all, X_all, split=0.7, seed=0)
    me = ModelEvaluator()

    print('Vectorization time:', vec_time)
    print('Data matrix size:', X_all.shape)
    print(dp.label_dict, '\n')

    # LinearSVC (liblinear SVM implementation, one-v-all)
    cross_validate = True
    if cross_validate:
        model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,\
            C=1, multi_class='ovr', fit_intercept=True, intercept_scaling=1,\
            class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
        param_grid = {'C':np.logspace(-2,2,24).tolist()}
        grid_info, grid_best, grid_time = me.param_search(model, param_grid,\
                y_train, X_train, num_folds=10)
        C = grid_best['C']
    else:
        C = 1

    SVM = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,\
            C=C, multi_class='ovr', fit_intercept=True, intercept_scaling=1,\
            class_weight='balanced', verbose=0, random_state=None, max_iter=1000)

    SVM_y_pred, SVM_test_acc, SVM_test_prec, SVM_test_rec, SVM_test_time = me.test(SVM, y_test, X_test)
    me.print_scores(dp, SVM_test_acc, SVM_test_prec, SVM_test_rec)

    # Perform semisupervised learning
    ssl = SemiSupervisedLearner(SVM)
    ave_conf = np.mean(ssl.confidence_scores(SVM.decision_function(X_train)))
    y_working = ssl.loop_learning(X_unlab, y_train, X_train, num_loops=10, conf_thresh=ave_conf)

    SSL_y_pred, SSL_test_acc, SSL_test_prec, SSL_test_rec, SSL_test_time = me.test(SVM, y_test, X_test)
    me.print_scores(dp, SSL_test_acc, SSL_test_prec, SSL_test_rec)
