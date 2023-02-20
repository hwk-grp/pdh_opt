import numpy
import torch
import fnn as fnn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def calc_score(yld, k_d, yld_max=0.5, k_d_max=1.0):
    part1 = yld / yld_max
    part2 = 1.0 - k_d / k_d_max

    for i in range(0, yld.shape[0]):
        if part1[i] > 1.0:
            part1[i] = 1.0
        if part2[i] < 0.0:
            part2[i] = 0.0

    return 0.5 * (part1 + part2)


def train_fnn_ens(dataset_train_x, dataset_train_y, batch_size, n_models, init_lr, l2_coeff):
    dataset_train = TensorDataset(torch.tensor(dataset_train_x, dtype=torch.float),
                                  torch.tensor(dataset_train_y, dtype=torch.float))
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    models = list()
    for i in range(0, n_models):
        models.append(fnn.FNN(dataset_train_x.shape[1], 1).cpu())

    for i in range(0, n_models):
        model = models[i]
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=l2_coeff)
        criterion = torch.nn.L1Loss()

        for epoch in range(0, 1000):
            train_loss = fnn.train(model, data_loader_train, optimizer, criterion)
            print(epoch, train_loss)

        torch.save(model.state_dict(), "model.%d.pt"%i)

    return models


def test_fnn_ens(models, dataset_test_x, dataset_test_y):
    preds_ens = numpy.zeros(dataset_test_y.shape)
    dataset_test = TensorDataset(torch.tensor(dataset_test_x, dtype=torch.float),
                                 torch.tensor(dataset_test_y, dtype=torch.float))
    data_loader_test = DataLoader(dataset_test, batch_size=32)

    for i in range(0, len(models)):
        preds_ens += fnn.test(models[i], data_loader_test)

    return preds_ens / len(models)


def pred_ens(models, x):
    pred = 0

    with torch.no_grad():
        for m in models:
            pred += m(x).numpy()

    return pred / len(models)


def sel_init_pos(data, targets, k, obj_func):
    scores = obj_func(targets)
    idx_sel = numpy.argpartition(scores, k)[:k]

    return data[idx_sel, :]
