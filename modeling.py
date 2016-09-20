import time
import numpy as np
from scipy import sparse as sp
from matplotlib import pyplot as plt

from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.learning_curve import learning_curve

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
    def test(self, model, y_test, X_test, label_indices=None):

        t0 = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - t0
        test_accuracy = metrics.accuracy_score(y_test, y_pred)
        test_precision = metrics.precision_score(y_test, y_pred, labels=label_indices, average=None)
        test_recall = metrics.recall_score(y_test, y_pred, labels=label_indices, average=None)

        return y_pred, test_accuracy, test_precision, test_recall, test_time

    def top_3_acc(self, model, y_test, X_test):

        # get hyperplane distances
        hp_dists = model.decision_function(X_test)
        top_3 = np.argsort(-hp_dists, axis=1)[:3,:]
        top_3_bool = y_pred.apply_along_axis()

        return top_3_score

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

        if type(conf_thresh) == np.ndarray:
            # differing thresholds for classes
            add_set = np.zeros_like(y_pred, dtype=bool)
            for i in range(conf_thresh.shape[0]):
                add_set[(conf_scores > conf_thresh[i]) * (y_pred == i)] = True
        else:
            # same threshold for all classes
            add_set = conf_scores > conf_thresh

        #print(np.unique(y_working))
        y_working = np.concatenate((y_working, y_pred[add_set]))
        X_working = sp.vstack((X_working, X_unlab[add_set]))
        X_unlab = X_unlab[~add_set]

        return y_working, X_working, X_unlab

    """
    Iterates the model and data to grow the training set using soft labels
    """
    def loop_learning(self, X_unlab, y_train, X_train, label_indices, num_loops, conf_thresh):

        X_working = X_train.copy()
        y_working = y_train.copy()
        num_added = 0

        for i in range(num_loops):

            hp_dists = self.model.decision_function(X_unlab)
            y_pred_ind = np.argmax(hp_dists, axis=1).reshape(-1,1)
            y_pred = np.apply_along_axis(lambda x: label_indices[x], 1, y_pred_ind)
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

            self.model.fit(X_working, y_working)

        return y_working
