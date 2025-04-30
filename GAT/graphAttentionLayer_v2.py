import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayerV2(nn.Module):
    """
    Memory-efficient GATv2 layer, avoiding explicit concatenation
    Reference: https://arxiv.org/abs/2105.14491
    """

    def __init__(self, in_features, out_features, batch_size, dropout, alpha, concat=True):
        super(GraphAttentionLayerV2, self).__init__()
        self.dropout = dropout  # Dropout参数
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.alpha = alpha  # LeakyReLU参数
        self.concat = concat  # 是否进行ELU激活
        self.batch_size = batch_size  # Batch size

        # 定义可训练参数
        self.W = nn.Parameter(torch.empty(size=(batch_size, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_src = nn.Parameter(torch.empty(size=(batch_size, out_features, 1)))
        self.a_dst = nn.Parameter(torch.empty(size=(batch_size, out_features, 1)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: 节点特征矩阵 [Batch_size, N, in_features]
        adj: 邻接矩阵 [Batch_size, N, N]
        """
        batch_size, N, _ = h.size()

        # 线性变换
        Wh = torch.einsum('bni, bio -> bno', h, self.W)  # [Batch_size, N, out_features]

        # 分别计算源节点和目标节点的注意力分数
        e_src = torch.einsum('bni, bio -> bn', Wh, self.a_src)  # [Batch_size, N]
        e_dst = torch.einsum('bni, bio -> bn', Wh, self.a_dst)  # [Batch_size, N]

        # 广播加和
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # [Batch_size, N, N]

        # 应用LeakyReLU
        e = self.leakyrelu(e)

        # 应用邻接矩阵，屏蔽无连接的注意力分数
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 应用注意力权重到节点特征
        h_prime = torch.einsum('bij, bjn -> bin', attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + f' ({self.in_features} -> {self.out_features})'
