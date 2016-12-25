# coding: utf-8
#pca+svm with grid search

from __future__ import print_function

from sklearn import svm
from sklearn.model_selection import GridSearchCV

# GridSearchと一緒にSVMを実行する関数
def svm_gds(train_d, test_d, parameters = {'kernel':('rbf', ), 'C':[1,], 'gamma':[0.1,]}):
    clf = GridSearchCV(svm.SVC(), parameters, n_jobs = -1)
    clf.fit(train_d['data'], train_d['target'])

    print(clf.best_estimator_)
    test_pred = clf.predict(test_d['data'])
    return test_pred

# SVMを実行する関数
def svm_param(train_d, test_d, parameters={'kernel':'rbf', 'C':1, 'gamma':0.1}):
    clf = svm.SVC(kernel=parameters['kernel'], C=parameters['C'], gamma=parameters['gamma'])
    clf.fit(train_d['data'], train_d['target'])
    test_pred = clf.predict(test_d['data'])
    return test_pred
