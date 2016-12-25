# coding: utf-8
# plot result with some format

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

# 作図用の関数
## confusion_matrixを図として表す関数
def show_cf(confusions, n_labels):
    plt.clf()
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(False)
    plt.xticks(np.arange(n_labels))
    plt.yticks(np.arange(n_labels))
    plt.imshow(confusions, cmap=plt.cm.jet, interpolation='nearest');

    plt.xlim([-0.5, 1.5])
    plt.ylim([1.5, -0.5])

    for i, cas in enumerate(confusions):
        for j, count in enumerate(cas):
            if count > 0:
                xoff = .07 * len(str(count))
                plt.text(j-xoff, i+.2, int(count), fontsize=32, color='white')
    return 0

# ROCカーブの作図の関数
def show_roc(test_labels, test_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_labels, test_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.clf()
    plt.figure()
    lw = 2
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MNIST label=0 and 1')
    plt.legend(loc="lower right")
    plt.show()
    return 0

# t-SNEを実行して可視化時にデータの次元を削減する
def app_tSNE(data):
    model = TSNE(n_components=2, random_state=0)
    tsne_data = model.fit_transform(data)
    return tsne_data

# 散布図を作成する関数
def show_scatter(data, target, d_name='hoge', n_plot=None):
    plt.clf()
    c = ['#ff0000','#0000ff','#d16b16','#00984b','#0074bf',
            '#c30068','#6d1782', '#546474', '#244765', '#8f253b']
    c_markers = ['s', 'v', 'o', 'd', '*', '+', 'H', 'p', 'x', 'D']
    for i in range(len(list(set(target)))):
        feat = data[np.where(target == i)]
        if n_plot != None:
            feat = feat[:n_plot]
        plt.plot(feat[:, 0], feat[:, 1], c_markers[i], markeredgecolor=c[i], markerfacecolor='#ffffff', markeredgewidth=2, markersize=10)
    plt.legend(['0', '1','2','3','4','5','6','7','8','9'], numpoints=1, borderaxespad=0, bbox_to_anchor=(1.17, 1))
    plt.subplots_adjust(left=0.1, right=0.85)
    plt.savefig('{}_vector.png'.format(d_name))
    return 0
