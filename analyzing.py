import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams

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
    def class_hist(self, y, label_indices, label_names, print_hist=False, show_now=False):

        counts = np.bincount(y)
        counts = [counts[i] for i in range(len(counts)) if i in label_indices]

        x = np.arange(len(counts))
        ymax = max(counts)*1.1

        self.fig_num += 1
        plt.figure(self.fig_num, figsize=(12, 10))
        plt.bar(x, counts, align='center', width=0.5)
        plt.ylim(0, ymax)
        plt.xticks(x, label_names, rotation=45, ha='right')
        rcParams.update({'figure.autolayout': True, 'font.size': 20})

        if print_hist:
            print(counts)

        if show_now:
            plt.show()

        return

    """
    Plot scores of classes in a bar chart
    """
    def class_scores(self, scores, label_names, show_now=False):

        if type(scores) is tuple:
            prec, rec = scores
            scores = np.array([2/(1/prec[i] + 1/rec[i])\
                    for i in range(len(prec))])

        x = np.arange(len(label_names))
        ymax = max(scores)*1.1

        self.fig_num += 1
        plt.figure(self.fig_num, figsize=(12,10))
        plt.bar(x, scores, align='center', width=0.5)
        plt.ylim(0, ymax)
        plt.xticks(x, label_names, rotation=45, ha='right')
        rcParams.update({'figure.autolayout': True, 'font.size': 20})

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
    def important_feats(self, model, feat_names, label_names, num_feats=5):

        feat_ranking = np.argsort(-model.coef_, axis=1)
        if type(feat_names) is list:
            feat_names = np.array(feat_names)

        for i in range(len(label_names)):

            print(label_names[i])
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
