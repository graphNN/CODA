# graph-CODA training

import dgl
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

import random
import time
import argparse

from CODA_models import dgl_gat, dgl_gcn, dgl_sage, dgl_appnp, dgl_gin
from CODA_utils import graph_augmentation, load_data, class_matrix, load_data_webkb


def select_model(model_name, dropout, tem):
    if model_name=='GAT':
        model = dgl_gat(input_dim=features.shape[1], out_dim=args.hidden, num_heads=[2, 1],
                        num_classes=(int(labels.max()) + 1), dropout=dropout,
                        lga=args.lga, tem=tem)
    if model_name == 'GCN':
        model = dgl_gcn(input_dim=features.shape[1], nhidden=args.hidden, nclasses=(int(labels.max()) + 1), dropout=dropout,
                        lga=args.lga, tem=tem, nlayers=2)
    if model_name == 'SAGE':
        model = dgl_sage(input_dim=features.shape[1], nhidden=args.hidden, aggregator_type='gcn',  # mean, gcn, pool, lstm
                         nclasses=(int(labels.max()) + 1), dropout=dropout,
                        lga=args.lga, tem=tem)
    if model_name == 'GIN':
        model = dgl_gin(input_dim=features.shape[1], hidden=args.hidden, aggregator_type='max',  # mean  sum  max
                         classes=(int(labels.max()) + 1), dropout=dropout,
                        lga=args.lga, tem=tem)
    if model_name == 'APPNP':
        model = dgl_appnp(input_dim=features.shape[1], hidden=args.hidden,
                         classes=(int(labels.max()) + 1), dropout=dropout,
                        lga=args.lga, tem=tem, k=10, alpha=0.1)
    return model

parser = argparse.ArgumentParser()

parser.add_argument('--lga', type=bool, default=True, help='learnable graph attention')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')  # 0
parser.add_argument('--dataset', type=str, default='cora', help='cora citeseer pubmed wisconsin texas cornell')
parser.add_argument('--model', type=str, default='GCN', help='GAT, GCN, GIN, APPNP, SAGE')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')  # 200
parser.add_argument('--patience', type=int, default=20, help='Patience')  # 20
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Dropout rate (1 - keep probability).')

parser.add_argument('--s', type=int, default=49, help='')
parser.add_argument('--T0', type=float, default=2.9, help='T_O: Temperature coefficient')
parser.add_argument('--T1', type=float, default=0.5, help='T_C: Temperature coefficient')
parser.add_argument('--dropout0', type=float, default=0.5, help='d_0: Dropout rate (1 - keep probability).')
parser.add_argument('--dropout1', type=float, default=0.6, help='d_1: Dropout rate (1 - keep probability).')
parser.add_argument('--self_dropout0', type=float, default=0.5, help='d_2: Dropout rate (1 - keep probability).')
parser.add_argument('--self_dropout1', type=float, default=0.6, help='d_3: Dropout rate (1 - keep probability).')
parser.add_argument('--prior_model', type=str, default='grand', help='pre-GNNs: grand, gemo')

args = parser.parse_args()

device = torch.device("cuda:0")
dropout = [args.dropout0, args.dropout1, args.self_dropout0,args.self_dropout1]
tem = [args.T0, args.T1]

# the original graph
if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
    A, features, labels, train_mask, val_mask, test_mask, adj = load_data(args.dataset)
    g = dgl.from_scipy(adj)
    g = dgl.add_self_loop(g).to(device)

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
else:
    g, adj, _, features, labels, train_mask, val_mask, test_mask = load_data_webkb(args.dataset, f'./data/new_data/'
                                                                f'{args.dataset}/{args.dataset}_split_0.6_0.2_1.npz',
                                                                                   0.6, 0.2)
    g = dgl.add_self_loop(g).to(device)
    features = torch.tensor(features).to(device)
    labels = torch.tensor(labels).to(device)
    train_mask = torch.tensor(train_mask, dtype=torch.bool).to(device)
    val_mask = torch.tensor(val_mask, dtype=torch.bool).to(device)
    test_mask = torch.tensor(test_mask, dtype=torch.bool).to(device)


if args.prior_model == 'grand' or args.prior_model == 'gemo':
    pro_outputs = torch.load(f'./outputs/{args.prior_model}_{args.dataset}_output.pkl').cpu().to(device)
    cat_dis = torch.load(f'./category_matrix/{args.prior_model}_{args.dataset}_cat_dis.pkl')
else:
    pro_outputs = torch.load(f'./outputs/{args.prior_model}_{args.dataset}_output.pkl').cpu().to(device)
    # cat_dis = class_matrix(g, pro_outputs, test_mask, labels)
    cat_dis, labels = class_matrix(g, pro_outputs, test_mask, labels)

# train

# the auxiliary graph
new_g = graph_augmentation(cat_dis, args.s * (len(g.edges()[0]-len(g.nodes())))).to(device)
new_g = dgl.add_self_loop(new_g)
g_list = []
g_list.append(g)
g_list.append(new_g)

test_accs = []
best_accs = []
# args.seed = 9
for i in range(args.seed, args.seed + 10):
    # seed
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)

    model = select_model(args.model, dropout, tem)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)

    t0 = time.time()
    best_acc, best_val_acc, best_test_acc, best_val_loss = 0, 0, 0, float("inf")
    for epoch in range(args.epochs):
        model.train()
        t1 = time.time()

        outputs = model(g_list, features)
        outputs_ = F.log_softmax(outputs, dim=1)
        train_loss = F.cross_entropy(outputs_[train_mask], labels[train_mask])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()  # val
        with torch.no_grad():
            outputs = model(g_list, features)
            outputs_ = F.log_softmax(outputs, dim=1)

            train_loss_ = F.cross_entropy(outputs_[train_mask], labels[train_mask]).item()
            train_pred = outputs_[train_mask].max(dim=1)[1].type_as(labels[train_mask])
            train_correct = train_pred.eq(labels[train_mask]).double()
            train_correct = train_correct.sum()
            train_acc = (train_correct / len(labels[train_mask])) * 100

            val_loss = F.cross_entropy(outputs_[val_mask], labels[val_mask]).item()
            val_pred = outputs_[val_mask].max(dim=1)[1].type_as(labels[val_mask])
            correct = val_pred.eq(labels[val_mask]).double()
            correct = correct.sum()
            val_acc = (correct / len(labels[val_mask])) * 100

        model.eval()  # test
        with torch.no_grad():
            test_loss = F.cross_entropy(outputs_[test_mask], labels[test_mask]).item()
            test_pred = outputs_[test_mask].max(dim=1)[1].type_as(labels[test_mask])
            correct = test_pred.eq(labels[test_mask]).double()
            correct = correct.sum()
            test_acc = (correct / len(labels[test_mask])) * 100

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc
                bad_epoch = 0

                # torch.save(outputs, '/diskvdb/home/dr/duanrui_0110/three_distance/outputs/CODA_citeseer.pkl')

            else:
                bad_epoch += 1

        epoch_time = time.time() - t1
        if (epoch + 1) % 1 == 0:
            print('Epoch: {:3d}'.format(epoch), 'Train loss: {:.4f}'.format(train_loss_),
                      '|Train accuracy: {:.2f}%'.format(train_acc), '||Val loss: {:.4f}'.format(val_loss),
                      '||Val accuracy: {:.2f}%'.format(val_acc), '||Test loss:{:.4f}'.format(test_loss))

        if bad_epoch == args.patience:
            break

    _time = time.time() - t0
    print('\n', 'Test accuracy:', best_test_acc)
    print('Time of training model:', _time)
    print('End of the training !')
    print('-' * 100)

    test_accs.append(best_test_acc.item())


print(test_accs)
print(f'Average test accuracy and : {np.mean(test_accs)} ± {np.std(test_accs)}')
print('-' * 50)


