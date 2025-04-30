import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT.graphAttentionLayer_v2 import GraphAttentionLayerV2


class GATV2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, batch_size, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GATV2, self).__init__()
        self.dropout = dropout

        # 加入Multi-head机制
        self.attentions = [GraphAttentionLayerV2(nfeat, nhid, batch_size, dropout=dropout, alpha=alpha, concat=True) for
                           _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayerV2(nhid * nheads, nclass, batch_size, dropout=dropout, alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=2)
