import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    '''
    得到输入序列经过注意力机制之后的输出
    Attention(Q,K,V) = softmax((Q*K.T)/sqrt(d_q_k))*V
    Scaled Dot-Product Attention
    '''

    def __init__(self, d_k_q, attn_dropout=0.25):
        super().__init__()
        self.scale = np.power(d_k_q,0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        '''
        Q K V 的形状都是(batch,sentenceLength,d_q_k,d_v)
        :param mask: 在decoder中会用到
        :return:
        '''

        # 矩阵乘后，shape: (batch,sentenceLength,sentenceLength)
        # 相当于每一句有了自己的一个每个词与其他词的注意力矩阵（此时还没sofmax）
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.scale

        # decoder中使用
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)


        # softmax
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # 注意力矩阵乘V，得到每一句话的每一个词都经过了注意力处理后的表达的矩阵
        # shape: (batch, sentenceLength, d_v)
        output = torch.bmm(attn, v)

        return output


class Attention(nn.Module):
    '''
    注意力层：自注意力 / 编码解码注意力
    encoder-decoder attention or self-attention
    '''

    def __init__(self,d_model,d_q_k,d_v,dropoutRate = 0.25):
        '''
        通过点乘 w_q w_k w_v 变成 Q K V
        :param d_model: 输入向量的长度（例如嵌入层之后是词向量的长度）
        :param d_k: key的维度
        :param d_v: value的维度
        :param dropoutRate: dropout的概率
        '''
        super(Attention,self).__init__()

        self.d_q_k = d_q_k
        self.d_v = d_v

        # 因为Q*K.T 所以两者第二维肯定一样
        # 三个矩阵用于生成Q K V
        self.w_q = nn.Linear(d_model, self.d_q_k)
        self.w_k = nn.Linear(d_model, self.d_q_k)
        self.w_v = nn.Linear(d_model, self.d_v)
        # 初始化这三个矩阵
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_q_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_q_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(d_k_q=self.d_q_k)


    def forward(self, q_input, k_input, v_input, mask=None):
        '''
        输入应该是(batch,sentenceLength,d_model/上一层的d_v)
        :param q_input: 乘w_q的张量，比如encoder中第一层就是词嵌入的矩阵
        :param k_input: 同上
        :param v_input: 同上
        :param mask:
        :return:
        '''
        d_q_k, d_v = self.d_q_k, self.d_v

        # 对应的输入分别乘三个w_q w_k w_v矩阵后得到 Q、K、V三个矩阵
        # 形状应该是(batch, sentenceLength, d_q_k/d_v)
        Q = self.w_q(q_input)
        K = self.w_k(k_input)
        V = self.w_v(v_input)

        # return shape: (batch, sentenceLength, d_v)
        output= self.attention(Q, K, V, mask=mask)

        return output