import numpy
import pandas
import torch
import fnn
from torch import nn

from artificial_bee_colony import ABC
from sklearn.preprocessing import scale

import sys

num_feats = 26
nComp = 20
nLimit = 3

dataset_raw = numpy.array(pandas.read_csv('propene_rxn_feat.v2.csv'))
dataset_raw = dataset_raw[:,:num_feats]

model = fnn.FNN(num_feats, 1).cpu()
model.load_state_dict(torch.load(sys.argv[1]))
model.eval()

def pnt_func(x):
    pnt = 0

    for i in range(0, x.shape[1]):
        if x[0, i] < -1.0:
            pnt -= 1e3

        if x[0, i] > 1.0:
            pnt -= 1e3

    sum_at_feat = torch.sum(x[0,:nComp])
    if numpy.abs(sum_at_feat - 1.0) > 0.05:
        pnt -= 1e3

    non_neg = 0
    for i in range(nComp):
        if x[0, i] < 0.0:
            pnt -= 1e3

        if numpy.abs(x[0, i]) > 0.0001:
            non_neg += 1

    # of cat comp < nLimit, 1 is for support
    if non_neg > (1+nLimit):
        pnt -= 1e3

    return pnt

# heuristic search
lbs = numpy.ones([num_feats])*(-1.0)
ubs = numpy.ones([num_feats])*(1.0)
opt = ABC(num_feats, model, lbs, ubs, opt_type='max', size_pop=100, lim_trial=0.001, pnt_func=pnt_func, dataset_raw=dataset_raw)
sol, val = opt.run(10000)

numpy.savetxt('sol.csv', sol, delimiter=',')

print(sol)
dataset = numpy.vstack((dataset_raw, sol))
dataset = scale(dataset)
sol = dataset[-1,:]
print(torch.exp(model(torch.tensor(sol, dtype=torch.float).view(1, -1))) - 0.001)

