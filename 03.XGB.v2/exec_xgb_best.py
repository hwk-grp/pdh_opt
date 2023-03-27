import os
import sys
import numpy
import pandas
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale

from xgboost import XGBRegressor

n_feats = 26
seed = int(sys.argv[1])
numpy.random.seed(seed)

## load data
dataset = numpy.array(pandas.read_csv('propene_rxn_feat.v2.csv'))
idx_rand = numpy.random.permutation(dataset.shape[0])
n_train = int(0.8 * dataset.shape[0])

dataset_train_x = dataset[idx_rand[:n_train], :n_feats]
dataset_test_x = dataset[idx_rand[n_train:], :n_feats]

dataset_train_y_scr = dataset[idx_rand[:n_train], n_feats + 2].reshape(-1,1)
dataset_test_y_scr = dataset[idx_rand[n_train:], n_feats + 2].reshape(-1,1)

## load best parameters
best_hyp_model_path ='./best_params/score/'

model = joblib.load(os.path.join(best_hyp_model_path, f'best_model_xgb.pkl'))

# train prediction model for score
model = model.fit(dataset_train_x, dataset_train_y_scr.flatten())

# save model
result_path = './reference/'
os.makedirs(result_path, exist_ok=True)

joblib.dump(model, os.path.join(result_path, f'predictor.{seed}.sav'))

# organize y_pred, metrices

preds_scr = model.predict(dataset_test_x).reshape(-1, 1)

print(mean_absolute_error(dataset_test_y_scr, preds_scr), r2_score(dataset_test_y_scr, preds_scr),
        numpy.max(dataset_test_y_scr), preds_scr[numpy.argmax(dataset_test_y_scr)][0],
        dataset_test_y_scr[numpy.argmax(preds_scr)][0], numpy.max(preds_scr))

numpy.savetxt(os.path.join(result_path, f'results.{seed}.csv'), 
                                           numpy.hstack([idx_rand[n_train:].reshape(-1, 1),
                                           dataset_test_y_scr,
                                           preds_scr]), delimiter=',')

