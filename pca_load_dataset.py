# coding: utf-8
# load dataset; MNIT and Beer dataset and apply one to PCA

import numpy as np
import six.moves.cPickle as pickle
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata

# データセットの取得とPCA適用の関数
# MNISTデータを取得．0, 1のラベルのデータのみ取り出す．
def load_mnist(n_labels=None):
    dataset = fetch_mldata('MNIST original')
    dataset['data'] = dataset['data'].reshape((
        len(dataset['data']), dataset['data'][0].size))
    n_train = 60000
    train_dataset = {'data': dataset['data'][:n_train] /255.,
                     'target': dataset['target'][:n_train]}
    test_dataset = {'data': dataset['data'][n_train:] /255.,
                    'target': dataset['target'][n_train:]}

    # 0, 1の２クラス分のデータのみ取り出す
    train_indexes = np.where(train_dataset['target'] < 2)[0]
    test_indexes = np.where(test_dataset['target'] < 2)[0]
    train_dataset = {'data': train_dataset['data'][train_indexes],
                    'target': train_dataset['target'][train_indexes]}
    test_dataset = {'data': test_dataset['data'][test_indexes],
                    'target': test_dataset['target'][test_indexes]}

    return train_dataset, test_dataset


def load_beer(N_TRAIN=500):
    beer_shape = (659, 32 * 32 * 3)
    dataset_dir = './all_imgs_dic.pkl'

    # データセットの読み込み
    dataset = label_data = pickle.load(open(dataset_dir))
    dataset['data'] = dataset['data'].reshape(beer_shape)
    rdm = np.random.permutation(len(dataset['target']))

    train_dataset = {'data': dataset['data'][rdm][:N_TRAIN],
        'target': dataset['target'][rdm][:N_TRAIN]}

    test_dataset = {'data': dataset['data'][rdm][N_TRAIN:],
        'target': dataset['target'][rdm][N_TRAIN:]}

    return train_dataset, test_dataset


def app_pca(dataset, data_dim=2):
    # data shape is 2 axises.(a number of data, data_dim)
    data = dataset['data']
    pca = PCA(n_components = data_dim)
    app_pca = pca.fit(data).transform(data)
    app_dataset = {'data': app_pca, 'target': dataset['target']}
    return app_dataset
