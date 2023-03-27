import numpy
import pandas
import joblib
from artificial_bee_colony import ABC
import sys

predictor = joblib.load(sys.argv[1])

num_feats = 26
nComp = 20
nLimit = 3

def pnt_func(x):
    pnt = 0

    for i in range(0, x.shape[1]):
        if x[0, i] < -1.0:
            pnt -= 1e3

        if x[0, i] > 1.0:
            pnt -= 1e3
            
    sum_at_feat = numpy.sum(x[0,:nComp])
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
opt = ABC(num_feats, predictor.predict, lbs, ubs, opt_type='max', size_pop=100, lim_trial=0.001, pnt_func=pnt_func)
sol, val = opt.run(10000)

numpy.savetxt('sol.csv', sol, delimiter=',')

print(sol)
print(predictor.predict(sol.reshape(1,-1)))
