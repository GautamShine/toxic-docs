#!/usr/bin/env python

"""
__author__ = 'Gautam Shine'
__email__ = 'gshine@stanford.edu'

Document classifier for the "Toxic Docs" repository from Columbia University
and the Center for Public Integrity. Data set consists of PDF files of
emails, memos, advertisements, news articles, scientific articles cited in
legal cases involving allegations of environmental harm from toxic substances.

"""

from processing import *
from modeling import *
from analyzing import *

import time
import numpy as np
from sklearn.svm import LinearSVC

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
    vectorizer, X_all_ngram, feat_names = dp.vectorize(docs, min_df=5, max_ngram=2)
    vec_time = time.time() - t0

    # Replace regex labels with human labels
    y_all = np.loadtxt('labels.txt', dtype=np.int32)
    counts = np.bincount(y_all[y_all != -1])
    counts = [counts[i] for i in range(len(counts)) if i in dp.label_index_list]

    # Add extra features from ToxicDocs to n-gram data matrix
    key_list = ['num_pages']
    feats = dp.get_feats(docs, key_list)
    X_all = dp.stack_feats(X_all_ngram, feats)
    key_list.append('length')
    feat_names.extend(key_list)

    print('Vectorization time:', vec_time)
    print('Data matrix size:', X_all.shape)

    y_train, X_train, ind_train, y_test, X_test, ind_test, X_unlab, ind_unlab =\
            dp.split_data(y_all, X_all, split=0.7, seed=0)
    me = ModelEvaluator()

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
        splits = np.linspace(0.1, 0.9, 300)
        me.generate_learning_curve(SVM, X_train, y_train, splits)

    SVM_train_acc, SVM_train_time = me.train(SVM, y_train, X_train)
    SVM_y_pred, SVM_test_acc, SVM_test_prec, SVM_test_rec, SVM_test_time =\
            me.test(SVM, y_test, X_test, dp.label_index_list)
    me.print_scores(dp, SVM_test_acc, SVM_test_prec, SVM_test_rec)

    t_n_score, top_n_vec = me.top_n_acc(SVM, y_test, X_test, dp.label_index_list, n=3)
    print(top_n_score)

    # Perform semisupervised learning
    ssl = SemiSupervisedLearner(SVM)
    ave_conf = np.mean(ssl.confidence_scores(SVM.decision_function(X_train)))

    y_working = ssl.loop_learning(X_unlab, y_train, X_train, dp.label_index_list,\
            num_loops=1, conf_thresh=3*ave_conf)
    SSL_y_pred, SSL_test_acc, SSL_test_prec, SSL_test_rec, SSL_test_time =\
            me.test(SVM, y_test, X_test, dp.label_index_list)
    me.print_scores(dp, SSL_test_acc, SSL_test_prec, SSL_test_rec)
