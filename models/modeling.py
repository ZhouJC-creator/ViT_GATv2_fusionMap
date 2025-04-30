# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from GAT.gat_v2 import GATV2

import models.configs as configs

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# 用于基于输入特征图计算节点之间的相对坐标、距离、角度以及位置权重
class RelativeCoordPredictor(nn.Module):
    def __init__(self):
        super(RelativeCoordPredictor, self).__init__()

    def forward(self, x):
        """
        前向传播计算相对坐标、锚点、位置权重和最大索引

        参数:
        - x: 输入特征图，形状为 (N, C, H, W)

        返回:
        - relative_coord_total: 相对距离和角度 (N, S, 2)
        - basic_anchor: 锚点坐标 (N, 2)
        - position_weight: 位置权重 (N, S, S)
        - reduced_x_max_index: 最大响应索引 (N)
        """
        # 获取批量大小、通道数、高度、宽度
        N, C, H, W = x.shape

        # 将通道维度进行求和，合并12个头，得到单通道特征图 (N, H, W)
        mask = torch.sum(x, dim=1)
        size = H

        # 将 mask 转换为一维向量 (N, S)，S = H * W
        mask = mask.view(N, H * W)
        # 计算每个样本的阈值 (N, 1)
        thresholds = torch.mean(mask, dim=1, keepdim=True)
        # 大于阈值的为1，反之为0 (N, S)
        binary_mask = (mask > thresholds).float()
        # 将二值 mask 恢复为原始形状 (N, H, W)
        binary_mask = binary_mask.view(N, H, W)

        # 应用二值 mask 到原始特征图上，屏蔽无效区域
        masked_x = x * binary_mask.view(N, 1, H, W)
        # 调整形状并转置，使通道维度到最后 (N, S, C)
        masked_x = masked_x.view(N, C, H * W).transpose(1, 2).contiguous()  # (N, S, C)
        # torch.mean(masked_x, dim=-1): (32, 196, 12) -> (32, 196)
        # torch.max(torch.mean(masked_x, dim=-1), dim=-1): (32, 196) -> (32)
        # 对每个位置求平均特征响应值 (N, S)
        _, reduced_x_max_index = torch.max(torch.mean(masked_x, dim=-1), dim=-1)

        # 构建样本索引 (N)
        basic_index = torch.from_numpy(np.array([i for i in range(N)])).cuda()

        # 构建基础坐标标签 (H, W, 2)
        basic_label = torch.from_numpy(self.build_basic_label(size)).float()

        # 将基础标签扩展到批量维度，并转换形状 (N, S, 2)
        label = basic_label.cuda()
        label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, H * W, 2)  # (N, S, 2)

        # 转换索引为长整型
        basic_index = basic_index.long()
        reduced_x_max_index = reduced_x_max_index.long()

        # 获取每个样本的锚点坐标 (N, 1, 2)
        basic_anchor = label[basic_index, reduced_x_max_index, :].unsqueeze(1)  # (N, 1, 2)
        # 计算相对坐标 (N, S, 2)
        relative_coord = label - basic_anchor
        # 归一化到特征图大小范围内
        relative_coord = relative_coord / size

        # 计算相对距离 (N, S)
        relative_dist = torch.sqrt(torch.sum(relative_coord ** 2, dim=-1))  # (N, S)

        # 计算相对角度，值范围在(-π, π)之间 (N, S)
        relative_angle = torch.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])  # (N, S) in (-pi, pi)cdm
        # 归一化到(0, 1)之间
        relative_angle = (relative_angle / np.pi + 1) / 2  # (N, S) in (0, 1)

        # 应用二值 mask 到相对距离和角度上 (N, S)
        binary_relative_mask = binary_mask.view(N, H * W)
        relative_dist = relative_dist * binary_relative_mask
        relative_angle = relative_angle * binary_relative_mask

        # 去除多余维度，得到基础锚点坐标 (N, 2)
        basic_anchor = basic_anchor.squeeze(1)  # (N, 2)

        # 将相对距离和角度拼接在一起 (N, S, 2)
        relative_coord_total = torch.cat((relative_dist.unsqueeze(2), relative_angle.unsqueeze(2)), dim=-1)

        # 计算每个位置的平均权重 (N, S)
        position_weight = torch.mean(masked_x, dim=-1)
        # (32, 196, 1)
        position_weight = position_weight.unsqueeze(2)
        # 计算位置权重矩阵 (N, S, S)
        position_weight = torch.matmul(position_weight, position_weight.transpose(1, 2))

        # relative_coord_total：包含相对距离和角度。
        # basic_anchor：每个样本的锚点坐标。
        # position_weight：节点之间的权重矩阵。
        # reduced_x_max_index：最大特征响应的索引
        # (32, 196, 2), (32, 2), (32, 196, 196), (32)
        return relative_coord_total, basic_anchor, position_weight, reduced_x_max_index

    def build_basic_label(self, size):
        basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
        return basic_label


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, batch_size, dropout):
        """
        GCN 模型的构造函数

        参数:
        - nfeat: 输入特征的维度（每个节点的特征数）
        - nhid: 隐层特征维度
        - nclass: 输出类别的数量（例如节点分类中的类别数）
        - dropout: Dropout的丢弃概率
        """
        super(GCN, self).__init__()
        # # 第一个图卷积层，将输入特征映射到隐层特征
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # # 第二个图卷积层，将隐层特征映射到最终的类别空间
        # self.gc2 = GraphConvolution(nhid, nclass)

        # self.gat = GAT(nfeat, nhid, nclass,batch_size, 0.2, 0.2, 2)
        self.gat = GATV2(nfeat, nhid, nclass, batch_size, 0.2, 0.2, 8)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        参数:
        - x: 输入特征矩阵，形状为 (N, nfeat)，N是节点数
        - adj: 邻接矩阵，形状为 (N, N)

        返回:
        - 输出特征矩阵，形状为 (N, nclass)
        """
        # # (N, nhid)
        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout)
        # # (N, nclass)
        # x = self.gc2(x, adj)
        x = self.gat(x, adj)
        return x


class Part_Structure(nn.Module):
    def __init__(self, config):
        super(Part_Structure, self).__init__()
        self.relative_coord_predictor = RelativeCoordPredictor()
        self.gcn = GCN(2, 512, config.hidden_size, config.batch_size, dropout=0.1)

    def forward(self, hidden_states, attention_map):
        """
        参数:
        - hidden_states: 输入的隐藏状态特征 (B, T, D)
        - attention_map: 注意力映射 (B, C, H, W)

        返回:
        - hidden_states: 更新后的隐藏状态
        """
        B, C, H, W = attention_map.shape
        # 通过相对坐标预测器计算结构信息
        # structure_info: (B, H*W, 2) 每个节点的相对坐标和角度
        # basic_anchor: (B, 2) 每个样本的锚点坐标
        # position_weight: (B, H*W, H*W) 每对节点之间的位置权重
        # reduced_x_max_index: (B) 每个样本中最大响应的索引
        structure_info, basic_anchor, position_weight, reduced_x_max_index = self.relative_coord_predictor(
            attention_map)

        # 使用 GCN 处理结构信息，捕获节点之间的图结构信息 (B, 784, 768)
        structure_info = self.gcn(structure_info, position_weight)

        # 对每个样本，通过锚点索引，将结构信息与隐藏状态相加，完成特征融合。
        for i in range(B):
            # 计算每个样本中锚点对应的索引
            index = int(basic_anchor[i, 0] * H + basic_anchor[i, 1])
            # 将特定锚点的结构信息加到对应样本的隐藏状态上
            hidden_states[i, 0] = hidden_states[i, 0] + structure_info[i, index, :]
        # 更新后的隐藏状态，整合了结构信息和原始特征 (B, T, D) T: patch 的数量 D: 特征维度
        return hidden_states


class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):

        # (32, 12, 196)
        last_map = x[:, :, 0, 1:]

        # (32, 12), (32, 12)
        # max_value, max_inx = last_map.max(2)
        # 32, 12
        B, C = last_map.size(0), last_map.size(1)
        # 196
        patch_num = last_map.size(-1)
        # 14
        H = patch_num ** 0.5
        H = int(H)
        # (32, 12, 14, 14)
        attention_map = last_map.view(B, C, H, H)

        # (32, 12, 196), (32, 12), (32, 12), (32, 12, 14, 14)
        return last_map, attention_map

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * (
                        (img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=(config.slide_step, config.slide_step))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"] - 1):
            layer = Block(config)
            self.layer.append(layer)  # 移除深拷贝，假设Block无内部状态依赖
        self.part_layer = Block(config)
        self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.part_structure = Part_Structure(config)
        self.part_select = Part_Attention()

    def forward(self, hidden_states):
        B = hidden_states.shape[0]
        attn_weights = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)
        attn_weights = torch.stack(attn_weights, dim=0)  # (num_layers, B, heads, seq, seq)
        num_layers = attn_weights.size(0)
        # 向量化计算相似度矩阵
        flat_attn = attn_weights.flatten(start_dim=2)  # (num_layers, B, D)
        flat_attn = flat_attn.permute(1, 0, 2)  # (B, num_layers, D)
        flat_attn_norm = flat_attn / torch.norm(flat_attn, dim=2, keepdim=True)
        similarity_matrix = torch.bmm(flat_attn_norm, flat_attn_norm.transpose(1, 2))  # (B, L, L)

        # 动态阈值计算（移除Otsu）
        triu_mask = torch.triu(torch.ones(num_layers, num_layers, device=hidden_states.device), diagonal=1).bool()
        triu_mask = triu_mask.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
        sim_values = similarity_matrix[triu_mask].view(B, -1)  # (B, L*(L-1)/2)
        mean_sim = sim_values.mean(dim=1)  # (B,)
        std_sim = sim_values.std(dim=1)  # (B,)
        threshold = mean_sim + 1.5 * std_sim  # (B,)
        # 筛选高相似度层对
        threshold_expanded = threshold.view(B, 1, 1)
        above_threshold = (similarity_matrix > threshold_expanded) & triu_mask

        # 确定融合层范围
        fusion_attn_weights = []
        for b in range(B):
            pairs = torch.nonzero(above_threshold[b], as_tuple=False)
            if len(pairs) == 0:
                # 默认取最后一层
                fusion_map = attn_weights[-1, b]
            else:
                lows = pairs[:, 0]
                highs = pairs[:, 1]
                lowest = lows.min().item()
                highest = highs.max().item()
                if lowest >= highest:
                    fusion_map = attn_weights[lowest, b]
                else:
                    fusion_map = attn_weights[lowest, b]
                    for i in range(lowest + 1, highest + 1):
                        fusion_map = torch.matmul(attn_weights[i][b], fusion_map)
            fusion_attn_weights.append(fusion_map)
        fusion_attn_weights = torch.stack(fusion_attn_weights, dim=0)

        # 学习结构信息
        _, a_map = self.part_select(fusion_attn_weights)
        # hidden_states: (32, 197, 768)  a_map: (32, 12), (32, 12), (32, 12, 14, 14)
        hidden_states = self.part_structure(hidden_states, a_map)

        hidden_states, _ = self.part_layer(hidden_states)
        encoded = self.part_norm(hidden_states)

        return encoded


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        part_encoded = self.encoder(embedding_output)
        return part_encoded


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size)
        self.part_head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        tokens = self.transformer(x)
        logits = self.part_head(tokens[:, 0])

        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss, logits
        else:
            return logits

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.part_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.part_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
