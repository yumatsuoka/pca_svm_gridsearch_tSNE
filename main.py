# coding: utf-8
# run pca+svm with grid search by yuma

from __future__ import print_function

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#### ハイパーパラメータ #########
# dataset is 'mnist' or 'beer'.
w_dataset = 'mnist'

# if wanna use pca, true
pca = False
data_dim = 2

# a number of training data
N_TRAIN = 500

# if wanna use grid search, true
use_gds = False
parameters = {'kernel':('rbf', ), 'C':[0.001, 0.1, 1, 5, 10, 100], 'gamma':[0.001, 0.1, 1, 5, 10, 100]}\
    if use_gds == True else {'kernel':'rbf', 'C':5, 'gamma':0.001}
###############################

get_dataset = load_mnist if w_dataset == 'mnist' else load_beer
print("datasetの読み込み")
train_dataset, test_dataset = get_dataset(N_TRAIN)
print("train.shape, test.shape", train_dataset['target'].shape, test_dataset['data'].shape)
if pca == True:
    train_dataset = app_pca(train_dataset, data_dim)
    test_dataset = app_pca(test_dataset, data_dim)
    print("PCAed train.shape, test.shape", train_dataset['target'].shape, test_dataset['data'].shape)

# 2次元までPCAで次元圧縮した場合はデータ空間を可視化.3次元以上のデータはt-SNEを適用してから可視化.
show_scatter(train_dataset['data'], train_dataset['target'], n_plot=100)\
    if data_dim < 3 and pca == True\
    else show_scatter(app_tSNE(train_dataset['data'], train_dataset['target'], n_plot=100))

# SVMを実行
app_svm = (svm_gds)if(use_gds == True)else(svm_param)
test_pred = app_svm(train_dataset, test_dataset, parameters)
test_labels = test_dataset['target']

# 評価してみる
print(classification_report(test_labels, test_pred))
print(accuracy_score(test_labels, test_pred))
cfmx = confusion_matrix(test_labels, test_pred)
show_cf(cfmx, n_labels)
show_roc(test_labels, test_pred)
