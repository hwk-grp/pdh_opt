import os
import sys
import numpy
import pandas
import joblib
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

debug_run = True
n_feats = 26
seed = 1
numpy.random.seed(seed)

## hyp params
range_est = [200, 400, 600, 800, 1000, 1200]
range_depth = [7, 8, 9, 10]
range_subsample = [0.8, 0.9, 1]
range_colsample = [0.8, 0.9, 1]
range_lr = [0.1, 0.05]

if debug_run is True:
    range_est = [200, ]
    range_depth = [7, ]
    range_subsample = [0.8, ]
    range_colsample = [0.8, ]
    range_lr = [0.1, 0.05]

## load data
dataset = numpy.array(pandas.read_csv('propene_rxn_feat.v1.csv'))
# dataset[:, :n_feats] = scale(dataset[:, :n_feats])
idx_rand = numpy.random.permutation(dataset.shape[0])
n_train = int(0.8 * dataset.shape[0])

dataset_train_x = dataset[idx_rand[:n_train], :n_feats]
dataset_test_x = dataset[idx_rand[n_train:], :n_feats]

dataset_train_y_scr = dataset[idx_rand[:n_train], n_feats + 2].reshape(-1,1)
dataset_test_y_scr = dataset[idx_rand[n_train:], n_feats + 2].reshape(-1,1)

## build cv model
cv_xgb = GridSearchCV(
    XGBRegressor(random_state=seed, n_jobs = 2
                  ), cv=5, n_jobs = 2, verbose = 1,
    scoring={'mae' : 'neg_mean_absolute_error'}, refit = 'mae',
    return_train_score=True,                   
    param_grid={"n_estimators" : range_est,
                "max_depth": range_depth, 
                "subsample": range_subsample, 
                "colsample_bytree": range_colsample,
                "learning_rate": range_lr})

## result path
best_hyp_path ='./best_params/score/'
if debug_run is True:
    best_hyp_path ='./best_params_debug/score/'
    
os.makedirs(best_hyp_path, exist_ok=True)

## run cv_regressor for hyp opt 
org_stdout = sys.stdout
f = open(os.path.join(best_hyp_path, 'best_outputs.txt'), 'w')
sys.stdout = f

grid = cv_xgb
grid.fit(dataset_train_x, dataset_train_y_scr.flatten())

print("best cv_train : {:.3f}".format(grid.cv_results_['mean_train_mae'].max()))
print("best cv_test : {:.3f}".format(grid.cv_results_['mean_test_mae'].max()))
print("best refit rmse: {:.3f}".format(grid.best_score_))
print("test rmse: {:.3f}".format(grid.score(dataset_test_x, dataset_test_y_scr.flatten())))
print("best_params_ : ", grid.best_params_)

pandas.DataFrame(grid.cv_results_).to_csv(os.path.join(best_hyp_path,'Grid_search_results.csv'))
with open(os.path.join(best_hyp_path, 'best_params.p'), 'wb') as fm:
    pickle.dump(grid.best_params_, fm)

sys.stdout = org_stdout
f.close()

## rerun single regressor with best cv_hypparams
best_params = grid.best_params_
model = XGBRegressor(random_state=seed, n_jobs=4, verbosity=1, **best_params)
model = model.fit(dataset_train_x, dataset_train_y_scr.flatten())

joblib.dump(model, os.path.join(best_hyp_path, f'best_model_xgb.pkl'))
print('model training.. Done')