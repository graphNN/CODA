# graph-CODA models

import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv, GraphConv, GINConv, APPNPConv, SAGEConv


class dgl_gat(nn.Module):
    def __init__(self, input_dim, out_dim, num_heads, num_classes, dropout, lga, tem):
        super(dgl_gat, self).__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout = dropout

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(num_heads[0] * out_dim)

        self.lga = lga
        self.ga11 = nn.Linear(2 * num_heads[0] * out_dim, 1)
        self.ga12 = nn.Linear(2 * num_heads[0] * out_dim, 1)
        self.ga21 = nn.Linear(2 * num_classes, 1)
        self.ga22 = nn.Linear(2 * num_classes, 1)

        self.tem = tem

        self.layer1 = GATConv(input_dim, out_dim, num_heads[0], feat_drop=dropout[0], attn_drop=dropout[1],
                              activation=None, allow_zero_in_degree=True, negative_slope=1.)
        self.layer2 = GATConv(num_heads[0] * out_dim, num_classes, num_heads[1], feat_drop=dropout[0], attn_drop=dropout[1],
                              activation=None, allow_zero_in_degree=True, negative_slope=1.)

        self.layer = nn.ModuleList()
        self.layer.append(self.layer1)
        self.layer.append(self.layer2)
        self.ga = nn.ModuleList()
        self.ga.append(self.ga11)
        self.ga.append(self.ga12)
        self.ga.append(self.ga21)
        self.ga.append(self.ga22)
        self.params1 = list(self.layer.parameters())
        self.params2 = list(self.ga.parameters())

    def forward(self, graph_list, input_features):
        if len(graph_list) == 1:
            g1 = graph_list[0].to(self.device)
            if self.use_bn:
                input_features = self.bn1(input_features)
            x1 = self.layer1(g1, input_features)  # input_dim * num_heads[0] * out_dim
            x1 = x1.flatten(1)
            x1 = F.elu(x1)

            x1 = self.layer2(g1, x1)
            x1 = x1.squeeze(1)
            if self.num_heads[1] > 1:
                x1 = torch.mean(x1, dim=1)

            x1 = F.elu(x1)

            return x1

        if len(graph_list) == 2:
            g1 = graph_list[0]
            g2 = graph_list[1]

            x11 = self.layer1(g1, input_features)
            x12 = self.layer1(g2, input_features)
            x11 = x11.flatten(1)
            x12 = x12.flatten(1)

            if self.lga:
                x_cat1 = torch.cat([x11, x12], dim=1)
                ga11 = self.ga11(x_cat1)
                ga11 = F.softmax(ga11, dim=1)
                ga11 = F.dropout(ga11, self.dropout[3], self.training)
                x12 = self.tem[1] * torch.diag(ga11) * x12

                x_cat2 = torch.cat([x12, x11], dim=1)
                ga12 = self.ga12(x_cat2)
                ga12 = F.softmax(ga12, dim=1)
                ga12 = F.dropout(ga12, self.dropout[2], self.training)
                x11 = self.tem[0] * torch.diag(ga12) * x11

            x1 = x11 + x12

            x21 = self.layer2(g1, x1)
            x21 = x21.squeeze(1)

            if self.num_heads[1] > 1:
                x21 = torch.mean(x21, dim=1)

            x22 = self.layer2(g2, x1)
            x22 = x22.squeeze(1)
            if self.num_heads[1] > 1:
                x22 = torch.mean(x22, dim=1)

            if self.lga:
                x_cat1 = torch.cat([x21, x22], dim=1)
                ga21 = self.ga21(x_cat1)
                ga21 = F.softmax(ga21, dim=1)
                ga21 = F.dropout(ga21, self.dropout[3], self.training)
                x22 = self.tem[1] * torch.diag(ga21) * x22

                x_cat2 = torch.cat([x22, x21], dim=1)
                ga22 = self.ga22(x_cat2)
                ga22 = F.softmax(ga22, dim=1)
                ga22 = F.dropout(ga22, self.dropout[2], self.training)
                x21 = self.tem[0] * torch.diag(ga22) * x21

            x2 = x21 + x22

            return x2


class dgl_gcn(nn.Module):
    def __init__(self, input_dim, nhidden, nclasses, lga, tem, dropout, nlayers):
        super(dgl_gcn, self).__init__()
        if nlayers == 2:
            self.layer1 = GraphConv(in_feats=input_dim, out_feats=nhidden, allow_zero_in_degree=True)
            self.layer2 = GraphConv(in_feats=nhidden, out_feats=nclasses, allow_zero_in_degree=True)
        else:
            self.layer = nn.ModuleList()
            for i in range(nlayers):
                if i == 0:
                    self.layer.append(GraphConv(input_dim, nhidden))
                elif i < nlayers - 1:
                    self.layer.append(GraphConv(nhidden, nhidden))
                else:
                    self.layer.append(GraphConv(nhidden, nclasses))

        self.lga = lga
        self.tem = tem
        self.dropout = dropout
        self.nlayers = nlayers

        self.ga11 = nn.Linear(2 * nhidden, 1)
        self.ga12 = nn.Linear(2 * nhidden, 1)
        self.ga21 = nn.Linear(2 * nclasses, 1)
        self.ga22 = nn.Linear(2 * nclasses, 1)

    def forward(self, g_list, features):
        if len(g_list) == 1:
            g = g_list[0]
            if self.nlayers == 2:
                x = self.layer1(g, features)
                x = self.layer2(g, x)
            else:
                x = features
                for i in range(self.nlayers):
                    x = self.layer[i](g, x)
            return x

        if len(g_list) == 2:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            x11 = self.layer1(g1, features)
            x12 = self.layer1(g2, features)

            if self.lga:
                x_cat1 = torch.cat([x11, x12], dim=1)
                ga11 = self.ga11(x_cat1)
                ga11 = F.softmax(ga11, dim=1)
                ga11 = F.dropout(ga11, self.dropout[3], self.training)
                x12 = self.tem[1] * torch.diag(ga11) * x12

                x_cat2 = torch.cat([x12, x11], dim=1)
                ga12 = self.ga12(x_cat2)
                ga12 = F.softmax(ga12, dim=1)
                ga12 = F.dropout(ga12, self.dropout[2], self.training)
                x11 = self.tem[0] * torch.diag(ga12) * x11

            x1 = x11 + x12

            x21 = self.layer2(g1, x1)
            x22 = self.layer2(g2, x1)

            if self.lga:
                x_cat1 = torch.cat([x21, x22], dim=1)
                ga21 = self.ga21(x_cat1)
                ga21 = F.softmax(ga21, dim=1)
                ga21 = F.dropout(ga21, self.dropout[3], self.training)
                x22 = self.tem[1] * torch.diag(ga21) * x22

                x_cat2 = torch.cat([x22, x21], dim=1)
                ga22 = self.ga22(x_cat2)
                ga22 = F.softmax(ga22, dim=1)
                ga22 = F.dropout(ga22, self.dropout[2], self.training)
                x21 = self.tem[0] * torch.diag(ga22) * x21
            x2 = x21 + x22

            return x2


class dgl_sage(nn.Module):
    def __init__(self, input_dim, nhidden, aggregator_type, nclasses, lga, tem, dropout):
        super(dgl_sage, self).__init__()
        self.layer1 = SAGEConv(in_feats=input_dim, out_feats=nhidden, aggregator_type=aggregator_type)
        self.layer2 = SAGEConv(in_feats=nhidden, out_feats=nclasses, aggregator_type=aggregator_type)

        self.lga = lga
        self.tem = tem
        self.dropout = dropout

        self.ga11 = nn.Linear(2 * nhidden, 1)
        self.ga12 = nn.Linear(2 * nhidden, 1)
        self.ga21 = nn.Linear(2 * nclasses, 1)
        self.ga22 = nn.Linear(2 * nclasses, 1)

    def forward(self, g_list, features):
        if len(g_list) == 1:
            g = g_list[0]
            x = self.layer1(g, features)
            x = self.layer2(g, x)
            return x

        if len(g_list) == 2:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            x11 = self.layer1(g1, features)
            x12 = self.layer1(g2, features)

            if self.lga:
                x_cat1 = torch.cat([x11, x12], dim=1)
                ga11 = self.ga11(x_cat1)
                ga11 = F.softmax(ga11, dim=1)
                ga11 = F.dropout(ga11, self.dropout[3], self.training)
                x12 = self.tem[1] * torch.diag(ga11) * x12

                x_cat2 = torch.cat([x12, x11], dim=1)
                ga12 = self.ga12(x_cat2)
                ga12 = F.softmax(ga12, dim=1)
                ga12 = F.dropout(ga12, self.dropout[2], self.training)
                x11 = self.tem[0] * torch.diag(ga12) * x11

            x1 = x11 + x12

            x21 = self.layer2(g1, x1)
            x22 = self.layer2(g2, x1)

            if self.lga:
                x_cat1 = torch.cat([x21, x22], dim=1)
                ga21 = self.ga21(x_cat1)
                ga21 = F.softmax(ga21, dim=1)
                ga21 = F.dropout(ga21, self.dropout[3], self.training)
                x22 = self.tem[1] * torch.diag(ga21) * x22

                x_cat2 = torch.cat([x22, x21], dim=1)
                ga22 = self.ga22(x_cat2)
                ga22 = F.softmax(ga22, dim=1)
                ga22 = F.dropout(ga22, self.dropout[2], self.training)
                x21 = self.tem[0] * torch.diag(ga22) * x21

            x2 = x21 + x22

            return x2


class dgl_appnp(nn.Module):
    def __init__(self, input_dim, hidden, classes, k, alpha, lga, tem, dropout):
        super(dgl_appnp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.layer1 = APPNPConv(k=k, alpha=alpha, edge_drop=0.5)
        self.layer2 = APPNPConv(k=k, alpha=alpha, edge_drop=0.5)
        self.set_parameters()

        self.lga = lga
        self.tem = tem
        self.dropout = dropout

        self.ga11 = nn.Linear(2 * hidden, 1)
        self.ga12 = nn.Linear(2 * hidden, 1)
        self.ga21 = nn.Linear(2 * classes, 1)
        self.ga22 = nn.Linear(2 * classes, 1)

    def set_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)

    def forward(self, g_list, features):
        features = self.fc1(features)
        if len(g_list) == 1:
            g = g_list[0]
            x = self.layer1(g, features)
            x = F.elu(self.fc2(x))
            x = self.layer2(g, x)
            x = F.elu(x)
            return x

        if len(g_list) == 2:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            x11 = self.layer1(g1, features)
            x11 = F.elu(x11)
            x12 = self.layer1(g2, features)
            x12 = F.elu(x12)

            if self.lga:
                x_cat1 = torch.cat([x11, x12], dim=1)
                ga11 = self.ga11(x_cat1)
                ga11 = F.softmax(ga11, dim=1)
                ga11 = F.dropout(ga11, self.dropout[3], self.training)
                x12 = self.tem[1] * torch.diag(ga11) * x12

                x_cat2 = torch.cat([x12, x11], dim=1)
                ga12 = self.ga12(x_cat2)
                ga12 = F.softmax(ga12, dim=1)
                ga12 = F.dropout(ga12, self.dropout[2], self.training)
                x11 = self.tem[0] * torch.diag(ga12) * x11

            x1 = x11 + x12

            x21 = self.layer2(g1, x1)
            x21 = self.fc2(F.elu(x21))
            x22 = self.layer2(g2, x1)
            x22 = self.fc2(F.elu(x22))

            if self.lga:
                x_cat1 = torch.cat([x21, x22], dim=1)
                ga21 = self.ga21(x_cat1)
                ga21 = F.softmax(ga21, dim=1)
                ga21 = F.dropout(ga21, self.dropout[3], self.training)
                x22 = self.tem[1] * torch.diag(ga21) * x22

                x_cat2 = torch.cat([x22, x21], dim=1)
                ga22 = self.ga22(x_cat2)
                ga22 = F.softmax(ga22, dim=1)
                ga22 = F.dropout(ga22, self.dropout[2], self.training)
                x21 = self.tem[0] * torch.diag(ga22) * x21

            x2 = x21 + x22

            return x2


class dgl_gin(nn.Module):
    def __init__(self, input_dim, hidden, classes, aggregator_type, lga, tem, dropout):
        super(dgl_gin, self).__init__()
        self.apply_func1 = nn.Linear(input_dim, hidden)
        self.apply_func2 = nn.Linear(hidden, classes)
        self.layer1 = GINConv(apply_func=self.apply_func1, aggregator_type=aggregator_type)
        self.layer2 = GINConv(apply_func=self.apply_func2, aggregator_type=aggregator_type)
        self.set_parameters()

        self.lga = lga
        self.tem = tem
        self.dropout = dropout

        self.ga11 = nn.Linear(2 * hidden, 1)
        self.ga12 = nn.Linear(2 * hidden, 1)
        self.ga21 = nn.Linear(2 * classes, 1)
        self.ga22 = nn.Linear(2 * classes, 1)

    def set_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.apply_func1.weight, gain=gain)
        nn.init.xavier_normal_(self.apply_func2.weight, gain=gain)

    def forward(self, g_list, features):
        if len(g_list) == 1:
            g = g_list[0]
            x = self.layer1(g, features)
            x = F.elu(x)
            x = self.layer2(g, x)
            x = F.elu(x)
            return x

        if len(g_list) == 2:
            g1 = g_list[0]
            g2 = g_list[1].to(g1.device)
            x11 = self.layer1(g1, features)
            x12 = self.layer1(g2, features)

            if self.lga:
                x_cat1 = torch.cat([x11, x12], dim=1)
                ga11 = self.ga11(x_cat1)
                ga11 = F.softmax(ga11, dim=1)
                ga11 = F.dropout(ga11, self.dropout[3], self.training)
                x12 = self.tem[1] * torch.diag(ga11) * x12

                x_cat2 = torch.cat([x12, x11], dim=1)
                ga12 = self.ga12(x_cat2)
                ga12 = F.softmax(ga12, dim=1)
                ga12 = F.dropout(ga12, self.dropout[2], self.training)
                x11 = self.tem[0] * torch.diag(ga12) * x11

            x1 = x11 + x12

            x21 = self.layer2(g1, x1)
            x22 = self.layer2(g2, x1)

            if self.lga:
                x_cat1 = torch.cat([x21, x22], dim=1)
                ga21 = self.ga21(x_cat1)
                ga21 = F.softmax(ga21, dim=1)
                ga21 = F.dropout(ga21, self.dropout[3], self.training)
                x22 = self.tem[1] * torch.diag(ga21) * x22

                x_cat2 = torch.cat([x22, x21], dim=1)
                ga22 = self.ga22(x_cat2)
                ga22 = F.softmax(ga22, dim=1)
                ga22 = F.dropout(ga22, self.dropout[2], self.training)
                x21 = self.tem[0] * torch.diag(ga22) * x21

            x2 = x21 + x22

            return x2