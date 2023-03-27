import numpy
import pandas
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from util import train_fnn_ens
from util import test_fnn_ens
from util import calc_score

import torch
import sys

n_feats = 26
batch_size = 32
n_models = 1
init_lr = 1e-3

numpy.random.seed(int(sys.argv[1]))
dataset = numpy.array(pandas.read_csv('propene_rxn_feat.v2.csv'))
dataset[:, :n_feats] = scale(dataset[:, :n_feats])
idx_rand = numpy.random.permutation(dataset.shape[0])
n_train = int(0.8 * dataset.shape[0])
dataset_train_x = dataset[idx_rand[:n_train], :n_feats]
dataset_test_x = dataset[idx_rand[n_train:], :n_feats]

# train prediction model for score
dataset_train_y_scr = dataset[idx_rand[:n_train], n_feats + 2].reshape(-1, 1)
dataset_test_y_scr = dataset[idx_rand[n_train:], n_feats + 2].reshape(-1, 1)
dataset_train_y_scr_ln = numpy.log(dataset_train_y_scr + 0.001)
dataset_test_y_scr_ln = numpy.log(dataset_test_y_scr + 0.001)

models_scr = train_fnn_ens(dataset_train_x, dataset_train_y_scr_ln, batch_size, n_models, init_lr, 1e-3)
preds_scr_ln = test_fnn_ens(models_scr, dataset_test_x, dataset_test_y_scr)
preds_scr = numpy.exp(preds_scr_ln) - 0.001

print(mean_absolute_error(dataset_test_y_scr, preds_scr), r2_score(dataset_test_y_scr, preds_scr),
        numpy.max(dataset_test_y_scr), preds_scr[numpy.argmax(dataset_test_y_scr)][0],
        dataset_test_y_scr[numpy.argmax(preds_scr)][0], numpy.max(preds_scr))

numpy.savetxt('results.csv', numpy.hstack([idx_rand[n_train:].reshape(-1, 1),
                                           dataset_test_y_scr,
                                           preds_scr]), delimiter=',')

