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
from sqlalchemy import create_engine
import psycopg2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize, scale
from sklearn.learning_curve import learning_curve

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

    def __init__(self, text_key, label_key, num_chars):
        self.text_key = text_key
        self.label_key = label_key
        self.nlp = NLP(num_chars)
        self.label_dict = {'email': 0, 'internal_memo': 1,
                'boardroom_minutes': 2, 'annual_report': 3,
                'public_relations': 4, 'general_correspondance': 5,
                'media': 6,'deposition': 7,
                'scientific_article_unpublished': 8,
                'scientific_article_published': 9,
                'advertisement': 10, 'trade_association': 11,
                'contract': 12, 'budget':13,
                'court_transcript':14, 'general_report':15,
                'not_english':18, 'misc':19, 'blank':20}
        self.inv_label_dict = {v: k for k, v in self.label_dict.items()}

    """
    Takes a bson and returns the corresponding list of dicts, the frequency of
    each label, and the number of unlabeled items
    """
    def load_bson(self, bson_file, plot_hist=False):

        # 'rb' for read as binary
        f = open(bson_file, 'rb')
        docs = bson.decode_all(f.read())

        labels = np.zeros((len(docs), 1))
        counts = defaultdict(int)

        for i in range(len(docs)):
            try:
                labels[i] = self.label_dict[docs[i][self.label_key]]
                counts[docs[i][self.label_key]] += 1
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
    Writes the document dump (list of dicts) to a PostgresSQL table
    """
    def write_to_db(self, docs, user, pw, host, db_name):

        db = create_engine('postgres://%s:%s@%s/%s'%\
                (user, pw, host, db_name))
        conn = psycopg2.connect(database=dbname, user=user)

        df = pd.DataFrame(docs)
        df['_id'] = df['_id'].map(str)
        df.to_sql('toxic_docs_table', engine)

        return

    """
    Applies a TF-IDF transformer and count vectorizer to the corpus to build
    n-gram features for classification
    """
    def vectorize(self, docs, min_df=2, max_ngram=2):

        docs = [x[self.text_key] for x in docs]
        vectorizer = TfidfVectorizer(min_df=min_df,\
                ngram_range=(1, max_ngram), tokenizer=self.nlp.tokenizer, sublinear_tf=True)

        return vectorizer, vectorizer.fit_transform(docs), vectorizer.get_feature_names()

    """
    Retrieves document features of the ToxicDocs collection
    """
    def get_feats(self, docs, key_list):

        feats = []
        for doc in docs:
            feats.append({k:v for k,v in doc.items() if k in key_list})
            feats[-1]['num_pages'] = 1+np.log(feats[-1]['num_pages'])
            feats[-1]['length'] = 1+np.log(len(doc[self.text_key]))

        vectorizer = DictVectorizer()
        X_feats = vectorizer.fit_transform(feats)
        X_feats = normalize(X_feats, axis=0, norm='max')

        return X_feats

    """
    Stacks extra features onto given data matrix
    """
    def stack_feats(self, X, feats):

        X = sp.hstack((X, feats))

        return X.tocsr()

    """
    Splits the labeled subset into train and test sets
    """
    def split_data(self, y_all, X_all, split=0.7, seed=0):
        
        indices = np.arange(y_all.shape[0])

        X_unlab = X_all[(y_all == -1).flatten()]
        ind_unlab = indices[(y_all == -1).flatten()]

        y_valid = y_all[y_all != -1]
        X_valid = X_all[(y_all != -1).flatten()]
        ind_valid = indices[(y_all != -1).flatten()]

        X_train, X_test, y_train, y_test, ind_train, ind_test =\
                train_test_split(X_valid, y_valid, ind_valid,\
                train_size=split, random_state=seed)

        return y_train, X_train, ind_train, y_test, X_test, ind_test, X_unlab, ind_unlab

    """
    Merges given classes into one label for cases when sample size is small
    """
    def merge_classes(self, merge_arr, y):

        y_merged = y.copy()
        merged_dict = {}

        for y_1, y_2 in merge_arr:
            pass

        return y_merged, merged_dict

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
    def test(self, model, y_test, X_test, labels=None):

        t0 = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - t0
        test_accuracy = metrics.accuracy_score(y_test, y_pred)
        test_precision = metrics.precision_score(y_test, y_pred, labels=labels, average=None)
        test_recall = metrics.recall_score(y_test, y_pred, labels=labels, average=None)

        return y_pred, test_accuracy, test_precision, test_recall, test_time

    """
    Traces a learning curve
    """
    def generate_learning_curve(self, model, X, y, splits=np.linspace(0.1,0.9,18), plot_curve=False):

        train_sizes, train_scores, test_scores = learning_curve(model, X, y, splits)

        if plot_curve:
            plt.figure()
            plt.plot(train_sizes, np.mean(test_scores, axis=1), color='red', linewidth=4)
            plt.show()

        return train_sizes, train_scores, test_scores

    """
    Utility function for printing class-wise precision/recall
    """
    def print_scores(self, dp, acc, prec, rec):

        scores = np.array([[2/(1/prec[i] + 1/rec[i]), prec[i], rec[i]] \
                for i in range(len(prec))])

        keys = list(dp.inv_label_dict.keys())

        print('Accuracy:', acc)
        print('Mean F1:', np.mean(scores[:,0]))
        for i in range(len(prec)):
            label = dp.inv_label_dict[keys[i]]
            print('{0} \n F1: {1:.3f}, P: {2:.3f}, R: {3:.3f}, '.format( \
                    label, scores[i][0], scores[i][1], scores[i][2]))
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

        if type(conf_thresh) is not float:
            # differing thresholds for classes
            add_set = np.zeros_like(y_pred, dtype=bool)
            for i in range(conf_thresh.shape[0]):
                add_set[(conf_scores > conf_thresh[i]) * (y_pred == i)] = True
        else:
            # same threshold for all classes
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
        num_added = 0

        for i in range(num_loops):

            self.model.fit(X_working, y_working)

            hp_dists = self.model.decision_function(X_unlab)
            y_pred = np.argmax(hp_dists, axis=1)
            conf_scores = self.confidence_scores(hp_dists)

            y_size_old = y_working.shape[0]
            y_working, X_working, X_unlab = self.propagate_labels(\
                    y_working, X_working, y_pred, X_unlab,\
                    conf_scores, conf_thresh=conf_thresh)

            if y_size_old == y_working.shape[0]:
                print('Converged with {0:d} added'.format(num_added))
                break

            num_added += y_working.shape[0] - y_size_old
            print(num_added, y_working.shape[0], X_unlab.shape[0])

        return y_working


"""
Utility functions for understanding and visualizing the data
"""
class DataAnalyzer():

    def __init__(self, text_key):
        self.text_key = text_key
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
    def class_hist(self, y, labels, print_hist, show_now=False):

        counts = np.bincount(y)

        x = np.arange(len(counts))
        ymax = max(counts)*1.1

        self.fig_num += 1
        plt.figure(self.fig_num)
        plt.bar(x, counts, align='center', width=0.5)
        plt.ylim(0, ymax)
        plt.xticks(x, labels, rotation=45, ha='right')
        rcParams.update({'figure.autolayout': True, 'font.size': 25})

        if print_hist:
            print(counts)

        if show_now:
            plt.show()

        return

    """
    Plot scores of classes in a bar chart
    """
    def class_scores(self, scores, labels, show_now=False):

        if type(scores) is tuple:
            prec, rec = scores
            scores = np.array([2/(1/prec[i] + 1/rec[i])\
                    for i in range(len(prec))])

        x = np.arange(len(labels))
        ymax = max(scores)*1.1

        self.fig_num += 1
        plt.figure(self.fig_num)
        print(x, x.shape, scores, scores.shape)
        plt.bar(x, scores, align='center', width=0.5)
        plt.ylim(0, ymax)
        plt.xticks(x, labels, rotation=45, ha='right')
        rcParams.update({'figure.autolayout': True, 'font.size': 25})

        if show_now:
            plt.show()

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
    Print a document
    """
    def print_doc(self, docs, doc_ind, num_chars=150):

        print('\n\n\n', doc_ind, '\n**********\n',\
            docs[doc_ind][self.text_key][:num_chars], '\n\n\n',\
            docs[doc_ind][self.text_key][-num_chars:])

        return


    """
    Randomly sample a few documents from a given class labeled correctly or not
    """
    def sample_docs(self, docs, y, y_pred, y_inds, target_y,\
            misclassified=None, num_docs=3, seed=0, num_chars=150):

        target_inds = (y == target_y)

        if misclassified is True:
            target_inds *= (y != y_pred)

        elif misclassified is False:
            target_inds *= (y == y_pred)

        np.random.seed(seed)
        sample_inds = np.random.choice(y_inds[target_inds], size=num_docs, replace=False)

        for i in range(num_docs):
            self.print_doc(docs, sample_inds[i], num_chars)

        return

    """
    Retrieves the most similar documents to a given target document using
    similarity measured by inner product in n-gram space
    """
    def similar_docs(self, docs, X, target_ind, num_docs=5, print_docs=True, num_chars=150):

        target_doc = X[target_ind, :].transpose()
        doc_sims = np.dot(X, target_doc)
        sim_ranking = np.argsort(-doc_sims.toarray().flatten())
       
        if print_docs:
            for i in range(num_docs):
                self.print_doc(docs, sim_ranking[i+1], num_chars)

        return sim_ranking[1:(num_docs+1)]


"""
Main
"""
if __name__ == '__main__':

    bson_file = 'documents.bson'
    label_key = 'document_type'
    text_key = 'text'

    # Process the raw data
    dp = DataProcessor(text_key, label_key, num_chars=300)
    da = DataAnalyzer(text_key)
    docs, y_all, counts = dp.load_bson(bson_file)
    t0 = time.time()
    vectorizer, X_all_ngram, feat_names = dp.vectorize(docs, min_df=2, max_ngram=2)
    vec_time = time.time() - t0

    # Add extra features from ToxicDocs to n-gram data matrix
    key_list = ['num_pages']
    feats = dp.get_feats(docs, key_list)
    X_all = dp.stack_feats(X_all_ngram, feats)

    y_train, X_train, ind_train, y_test, X_test, ind_test, X_unlab, ind_unlab =\
            dp.split_data(y_all, X_all, split=0.7, seed=0)
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

    plot_learning = True
    if plot_learning:
        splits = np.linspace(0.1, 0.9, 100)
        me.generate_learning_curve(SVM, X_train, y_train, splits)

    SVM_y_pred, SVM_test_acc, SVM_test_prec, SVM_test_rec, SVM_test_time = me.test(SVM, y_test, X_test)
    me.print_scores(dp, SVM_test_acc, SVM_test_prec, SVM_test_rec)

    # Perform semisupervised learning
    ssl = SemiSupervisedLearner(SVM)
    ave_conf = np.mean(ssl.confidence_scores(SVM.decision_function(X_train)))
    y_working = ssl.loop_learning(X_unlab, y_train, X_train, num_loops=10, conf_thresh=ave_conf)

    SSL_y_pred, SSL_test_acc, SSL_test_prec, SSL_test_rec, SSL_test_time = me.test(SVM, y_test, X_test)
    me.print_scores(dp, SSL_test_acc, SSL_test_prec, SSL_test_rec)
