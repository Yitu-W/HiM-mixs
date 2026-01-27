"""
 date:3.10
 author:yitu
整合模型代码

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# 自注意力机制 (Self-Attention) 类
class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        """
        初始化自注意力机制层。
        input_dim: 每个时间步输入的特征维度
        attention_dim: 注意力机制中用于计算Query, Key, Value的特征维度
        """
        super(SelfAttention, self).__init__()

        # 定义3个线性变换（Query, Key, Value）
        self.query_linear = nn.Linear(input_dim, attention_dim)
        self.key_linear = nn.Linear(input_dim, attention_dim)
        self.value_linear = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        """
        执行自注意力计算。
        x: 输入数据的形状为 [batch_size, seq_len, input_dim]
        返回:
        - attention_output: 加权后的输入，形状 [batch_size, seq_len, attention_dim]
        - attention_weights: 计算出的注意力权重，形状 [batch_size, seq_len, seq_len]
        """
        # 计算 Query, Key, Value
        Q = self.query_linear(x)  # 计算 Query, 形状: [batch_size, seq_len, attention_dim]
        K = self.key_linear(x)  # 计算 Key, 形状: [batch_size, seq_len, attention_dim]
        V = self.value_linear(x)  # 计算 Value, 形状: [batch_size, seq_len, attention_dim]

        # 计算注意力得分，利用点积来衡量 Query 和 Key 的相关性
        # scores 形状: [batch_size, seq_len, seq_len]，即每个时间步对其他时间步的相关性
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # 缩放点积

        # 使用 softmax 归一化注意力得分，确保权重和为 1
        attention_weights = F.softmax(scores, dim=-1)  # 形状: [batch_size, seq_len, seq_len]

        # 计算加权后的输出：加权每个时间步的 Value
        attention_output = torch.matmul(attention_weights, V)  # 形状: [batch_size, seq_len, attention_dim]

        return attention_output,attention_weights


# GRU 网络 类
class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = SelfAttention(input_size, input_size)
    def forward(self, x):
        #x=self.attention(x)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


# 双向门控（BIGRU） 类
class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=6):
        """
        初始化 BiGRU 模型。
        :param input_dim: 输入特征的维度
        :param hidden_dim: GRU 隐藏层的维度
        :param num_layers: GRU 层数
        """
        super(BiGRU, self).__init__()
        # 定义 BiGRU 层
        self.bigru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        """
        前向传播过程，返回 GRU 的输出
        :param x: 输入数据，形状: [batch_size, seq_len, input_dim]
        :return: GRU 输出，形状: [batch_size, seq_len, hidden_dim * 2]
        """
        gru_out, _ = self.bigru(x)
        return gru_out


# BiGRU 和自注意力机制结合的模型 类
class BiGRUWithSelfAttention(nn.Module): #继承自定义神经网络类
    def __init__(self, input_dim, hidden_dim, attention_dim, num_classes, num_layers=6):
        """
        初始化 BiGRU 和自注意力机制结合的模型。
        :param input_dim: 每个时间步的输入特征维度
        :param hidden_dim: GRU 隐藏层的维度
        :param attention_dim: 自注意力层的维度
        :param num_classes: 分类任务的类别数
        :param num_layers: GRU 层数，默认 1
        """
        super(BiGRUWithSelfAttention, self).__init__()
        # 定义 BiGRU 层
        self.bigru = BiGRU(input_dim, hidden_dim, num_layers)
        # 定义自注意力层
        self.attention = SelfAttention(hidden_dim * 2, attention_dim)  # 双向GRU输出是 hidden_dim*2
        # 定义分类层
        self.fc = nn.Linear(attention_dim, num_classes)

    def forward(self, x):
        """
        前向传播过程
        :param x: 输入数据，形状为 [batch_size, seq_len, input_dim]
        :return: 分类结果和注意力权重
        """
        gru_out = self.bigru(x)  # GRU 输出，形状: [batch_size, seq_len, hidden_dim*2]
        attention_out, attention_weights = self.attention(gru_out)  # 自注意力输出
        pooled_output = attention_out.mean(dim=1)  # 平均池化
        output = self.fc(pooled_output)  # 分类层输出
        return output



class BiGRUsa(nn.Module): #继承自定义神经网络类
    def __init__(self, input_dim, hidden_dim, attention_dim,num_layers=6):
        """
        初始化 BiGRU 和自注意力机制结合的模型。
        :param input_dim: 每个时间步的输入特征维度
        :param hidden_dim: GRU 隐藏层的维度
        :param attention_dim: 自注意力层的维度
        :param num_classes: 分类任务的类别数
        :param num_layers: GRU 层数，默认 1
        """
        super(BiGRUsa, self).__init__()
        # 定义 BiGRU 层
        self.bigru = BiGRU(input_dim, hidden_dim, num_layers)
        # 定义自注意力层
        self.attention = SelfAttention(hidden_dim * 2, attention_dim)  # 双向GRU输出是 hidden_dim*2


    def forward(self, x):
        """
        前向传播过程
        :param x: 输入数据，形状为 [batch_size, seq_len, input_dim]
        :return: 分类结果和注意力权重
        """
        gru_out = self.bigru(x)  # GRU 输出，形状: [batch_size, seq_len, hidden_dim*2]
        attention_out, attention_weights = self.attention(gru_out)  # 自注意力输出
        return attention_out
#----------------------------------------------------------------------------------------------------------

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers  # 新增层数参数

        # 多层LSTM时自动启用dropout（PyTorch要求num_layers>1时dropout才生效）
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,  # 仅多层时启用输入dropout
            num_layers=num_layers  # 新增层数配置
        )
        self.dropout = nn.Dropout(0.1)  # 循环状态dropout（独立于LSTM层的dropout）
        self.fc = nn.Linear(hidden_dim, output_dim)  # 全连接输出层

    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim) = (?, 20, 3)
        lstm_out, _ = self.lstm(x)  # lstm_out形状: (batch_size, seq_len, hidden_dim)
        last_timestep = lstm_out[:, -1, :]  # 取最后一个时间步的输出 (batch_size, hidden_dim)
        x = self.dropout(last_timestep)
        logits = self.fc(x)  # 输出形状: (batch_size, num_classes)
        return logits


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMNetwork1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNetwork1, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
#----------------------------------------------------------------------------------------------------------

# 定义 状态空间模型（SSM） 类
class SSMModel(nn.Module):
    def __init__(self, input_size, state_size, output_size):
        super(SSMModel, self).__init__()
        # 保存 state_size 为实例属性
        self.state_size = state_size
        # 状态转移矩阵 A
        self.A = nn.Linear(state_size, state_size)
        # 输入矩阵 B
        self.B = nn.Linear(input_size, state_size)
        # 观测矩阵 C
        self.C = nn.Linear(state_size, output_size)
        # 这里暂不使用 D 矩阵，可根据实际情况添加
        # self.D = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        # 初始化状态向量
        state = torch.zeros(batch_size, self.state_size).to(x.device)
        #print(f"Output state shape: {x.shape}")

        for t in range(seq_len):
            # 状态转移
            state = self.relu(self.A(state) + self.B(x[:, t, :]))

        #print(f"Output 转移后state shape: {state.shape}")  #500*128
        # 观测输出
        output = self.C(state)
        #print(f"Output output shape: {output.shape}")  #500*11
        return output



#----------------------------------------------------------------------------------------------------------
# 位置编码 类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):

        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + 0.1 * self.pe[:, :x.size(1)] #XIU


# 基于tranformer的分类模型 类
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=3, d_model=1024, nhead=8, num_encoder_layers=6,
                 dim_feedforward=4096, dropout=0.1, num_classes=15):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model) #JIA
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
            ),
            num_layers=num_encoder_layers
        )

        self.fc_out = nn.Linear(d_model, num_classes)  #线性层将 d_model 的特征投影到 num_classes 维度。

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.input_fc(src) * math.sqrt(self.d_model)
        src = self.norm(src)  #JIA
        src = self.pos_encoder(src)

        # 确保 Transformer 兼容 PyTorch 2.x 的 `forward()` 调用方式
        if src_mask is not None:
            output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = output.mean(dim=1)  # Global average pooling over the sequence length
        return self.fc_out(output)

#----------------------------------------------------------------------------------------------------------

# TSMixer的残差块 类
class ResBlock(nn.Module):
    """Residual block of TSMixer."""

    def __init__(self, input_shape, dropout, ff_dim):
        super(ResBlock, self).__init__()

        # Temporal Linear
        self.norm1 = nn.BatchNorm1d(input_shape[0] * input_shape[1])
        self.linear1 = nn.Linear(input_shape[0], input_shape[0])
        self.dropout1 = nn.Dropout(dropout)

        # Feature Linear
        self.norm2 = nn.BatchNorm1d(input_shape[0] * input_shape[1])
        self.linear2 = nn.Linear(input_shape[-1], ff_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.linear3 = nn.Linear(ff_dim, input_shape[-1])
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        inputs = x

        # Temporal Linear
        x = self.norm1(torch.flatten(x, 1, -1)).reshape(x.shape)
        x = torch.transpose(x, 1, 2)
        x = F.relu(self.linear1(x))
        x = torch.transpose(x, 1, 2)
        x = self.dropout1(x)

        res = x + inputs

        # Feature Linear
        x = self.norm2(torch.flatten(res, 1, -1)).reshape(res.shape)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.dropout3(x)

        return x + res


# https://github.com/ts-kim/RevIN/blob/master/RevIN.py
# TSMixer的正则化 类
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps * self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x


# TSMixer 类
class TSMixerRevIN(nn.Module):
    """Implementation of TSMixerRevIN."""

    def __init__(self, input_shape, num_classes, n_block, dropout, ff_dim, target_slice):
        super(TSMixerRevIN, self).__init__()

        self.target_slice = target_slice

        self.rev_norm = RevIN(input_shape[-1])

        self.res_blocks = nn.ModuleList([ResBlock(input_shape, dropout, ff_dim) for _ in range(n_block)])

        self.linear1 = nn.Linear(input_shape[0], num_classes)
        self.linear2 = nn.Linear(input_shape[0], input_shape[0])
        # 线性变换层，将 6 维特征映射到 1 维
        self.fc1 = nn.Linear(6, 1)  # 将最后一个维度映射到1
        self.fc2 = nn.Linear(256, 6)  # 将最后一个维度映射到1

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024, dropout=dropout, batch_first=True
            ),
            num_layers=6
        )

        self.attention1 = SelfAttention(6,6)
        self.attention2 = SelfAttention(6, 256)
    def forward(self, x):

        x = self.rev_norm(x, 'norm')   #500,20,6
        #print(f"Output rev_norm shape: {x.shape}")
        #增加注意力机制
        x = self.attention1(x) #500,20,6
        #print(f"Output attention1 shape: {x.shape}")
        #x=self.transformer_encoder(x)
        #print(f"Output encoder shape: {x.shape}")
        #x=self.fc2(x)
        #print(f"Output fc2 shape: {x.shape}")

        for res_block in self.res_blocks:
            x = res_block(x)            #500,20,6
            #x = self.attention1(x)      #500,20,6
       # print(f"Output block shape: {x.shape}")

        if self.target_slice:
            x = x[:, :, self.target_slice]
        #print(f"Output shape: {x.shape}")

        #x = self.attention2(x)  #500,20,20
        #print(f"Output shape: {x.shape}")
        x = torch.transpose(x, 1, 2)  #500,20,20
        #print(f"Output shape: {x.shape}")
        #x = self.linear2(x)       #500,20,20
        x = self.linear1(x)  # 500,20,9
        #print(f"Output shape: {x.shape}")
        x = torch.transpose(x, 1, 2)   #500,9,20
        #print(f"Output shape: {x.shape}")
        x = self.rev_norm(x, 'denorm', self.target_slice)
        #print(f"Output shape: {x.shape}")
        #x = x.mean(dim=2)  # 平均池化
        #x = self.fc1(x)  # 变成 [500, class, 1]
        #print(f"Output shape: {x.shape}")
        # 去掉最后一个维度，使其变成 [500, class]
        #x = x.squeeze(-1)
        #print(f"Output shape: {x.shape}")
        return x


#----------------------------------------------------------------------------------------------------------
class AttnPool(nn.Module):
    def __init__(self, c, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(c, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):  # x: [B, L, C]
        w = self.fc2(torch.tanh(self.fc1(x))).squeeze(-1)  # [B, L]
        a = torch.softmax(w, dim=1).unsqueeze(-1)          # [B, L, 1]
        return (x * a).sum(dim=1)                          # [B, C]


# MLP mixer 类
class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class MixerLayer(nn.Module):
    def __init__(self, seq_length, num_channels, token_mixer_hidden_dim, channel_mixer_hidden_dim):
        super(MixerLayer, self).__init__()
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(seq_length),
            MLPBlock(seq_length, token_mixer_hidden_dim)
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, channel_mixer_hidden_dim)
        )

    def forward(self, x):
        # Token - Mixing
        x = x + self.token_mixer(x.transpose(1, 2)).transpose(1, 2)
        # Channel - Mixing
        x = x + self.channel_mixer(x)
        return x

class M_mixs(nn.Module):
    def __init__(self, seq_length, input_channels, num_classes, num_channels, num_layers, token_mixer_hidden_dim,
                 channel_mixer_hidden_dim):
        super(M_mixs, self).__init__()
        self.input_projection = nn.Linear(input_channels, num_channels)
        self.mixer_layers = nn.ModuleList([
            MixerLayer(seq_length, num_channels, token_mixer_hidden_dim, channel_mixer_hidden_dim)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(num_channels)
        self.classification_head = nn.Linear(num_channels, num_classes)
        #self.self_attention = SelfAttention(num_channels, num_channels)

    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)
        # x,_ = self.self_attention(x)
        # print(x.shape)
        for layer in self.mixer_layers:
            x = layer(x)
        x = self.layer_norm(x)
        #x, _ = self.self_attention(x)
        # 全局平均池化
        x = x.mean(dim=1)
        x = self.classification_head(x)
        return x

class M_mixs_1(nn.Module):
    def __init__(self, seq_length, input_channels, num_channels, num_layers, token_mixer_hidden_dim,
                 channel_mixer_hidden_dim):
        super(M_mixs_1, self).__init__()
        self.input_projection = nn.Linear(input_channels, num_channels)
        self.mixer_layers = nn.ModuleList([
            MixerLayer(seq_length, num_channels, token_mixer_hidden_dim, channel_mixer_hidden_dim)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(num_channels)
        self.pool = AttnPool(num_channels)

    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)

        for layer in self.mixer_layers:
            x = layer(x)
        # 全局平均池化
        x = self.layer_norm(x)
        x = self.pool(x)
        return x

class LSTMNetwork1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNetwork1, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.pool = AttnPool(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.pool(out)
        out = self.fc(out)
        return out

class CNNLSTMSA1(nn.Module):
    """
    CNN -> LSTM -> Self-Attention -> (mean pool) -> FC
    输入:  x [B, T, input_dim]
    输出:  logits [B, num_classes]
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels=(64, 128),
        lstm_hidden=256,
        lstm_layers=3,
        bidirectional=False,
        attention_dim=256,
        dropout=0.1,
    ):
        super(CNNLSTMSA1,self).__init__()

        # ---- CNN(1D) 特征提取：卷积若干层，但只池化一次（对齐原文） ----
        cnn = []
        in_ch = input_dim
        for out_ch in conv_channels:
            cnn += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        # 只在最后加一次 pooling（window=2）
        cnn += [nn.MaxPool1d(kernel_size=2)]

        self.cnn = nn.Sequential(*cnn)

        # ---- LSTM：输入特征维 = CNN 输出通道数 ----
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---- Self-Attention：对 LSTM 输出序列做时序注意力 ----
        self.attn = SelfAttention(lstm_out_dim, attention_dim)
        self.pool = AttnPool(attention_dim)
        # ---- 分类头 ----
        self.head = nn.Sequential(
            nn.Linear(attention_dim, num_classes)
        )

    def forward(self, x, return_attn=False):
        """
        x: [B, T, input_dim]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, T, input_dim], got {tuple(x.shape)}")

        # CNN 需要 [B, C, T]
        x = x.permute(0, 2, 1)          # [B, input_dim, T]
        x = self.cnn(x)                 # [B, Cc, T']

        # LSTM 需要 [B, T', Cc]
        x = x.permute(0, 2, 1)          # [B, T', Cc]
        lstm_out, _ = self.lstm(x)      # [B, T', H(*dir)]

        attn_out, attn_w = self.attn(lstm_out)  # attn_out: [B, T', A], attn_w: [B, T', T']

        # 时间维池化（你也可以换成 max 或者取最后时刻）
        pooled = self.pool(attn_out)
        logits = self.head(pooled)      # [B, num_classes]

        if return_attn:
            return logits, attn_w
        return logits


# BiGRU 和自注意力机制结合的模型 类
class BiGRUWithSelfAttention1(nn.Module): #继承自定义神经网络类
    def __init__(self, input_dim, hidden_dim, attention_dim, num_classes, num_layers=6):
        """
        初始化 BiGRU 和自注意力机制结合的模型。
        :param input_dim: 每个时间步的输入特征维度
        :param hidden_dim: GRU 隐藏层的维度
        :param attention_dim: 自注意力层的维度
        :param num_classes: 分类任务的类别数
        :param num_layers: GRU 层数，默认 1
        """
        super(BiGRUWithSelfAttention1, self).__init__()
        # 定义 BiGRU 层
        self.bigru = BiGRU(input_dim, hidden_dim, num_layers)
        # 定义自注意力层
        self.attention = SelfAttention(hidden_dim * 2, attention_dim)  # 双向GRU输出是 hidden_dim*2
        # 定义分类层
        self.pool = AttnPool(attention_dim)
        self.fc = nn.Linear(attention_dim, num_classes)

    def forward(self, x):
        """
        前向传播过程
        :param x: 输入数据，形状为 [batch_size, seq_len, input_dim]
        :return: 分类结果和注意力权重
        """
        gru_out = self.bigru(x)  # GRU 输出，形状: [batch_size, seq_len, hidden_dim*2]
        attention_out, attention_weights = self.attention(gru_out)  # 自注意力输出
        pooled_output = self.pool(attention_out)
        output = self.fc(pooled_output)  # 分类层输出
        return output


class M_mixs_2(nn.Module):
    def __init__(self, seq_length, input_channels, num_classes, num_channels, num_layers, token_mixer_hidden_dim,
                 channel_mixer_hidden_dim):
        super(M_mixs_2, self).__init__()
        #self.input_projection = nn.Linear(input_channels, num_channels)
        self.mixer_layers = nn.ModuleList([
            MixerLayer(seq_length, num_channels, token_mixer_hidden_dim, channel_mixer_hidden_dim)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(num_channels)
        self.classification_head = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        # 输入投影
        for layer in self.mixer_layers:
            x = layer(x)
        # 全局平均池化
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.classification_head(x)
        return x


class himmixs5(nn.Module):
    def __init__(self,input_dim, output_size):
        super(himmixs5, self).__init__()
        self.mix1 =  M_mixs_1(seq_length=5, input_channels=input_dim, num_channels=128, num_layers=2, token_mixer_hidden_dim=128, channel_mixer_hidden_dim=128)
        self.mix2 = M_mixs_2(seq_length=16, input_channels=128, num_classes=output_size, num_channels=128, num_layers=3, token_mixer_hidden_dim=256, channel_mixer_hidden_dim=512)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1+ quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5+ quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9+ quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13+ quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1) # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16], dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

class himmixs2(nn.Module):
    def __init__(self,input_dim, output_size):
        super(himmixs2, self).__init__()
        self.mix1 =  M_mixs_1(seq_length=2, input_channels=input_dim, num_channels=128, num_layers=2, token_mixer_hidden_dim=128, channel_mixer_hidden_dim=128)
        self.mix2 = M_mixs_2(seq_length=19, input_channels=128, num_classes=output_size, num_channels=128, num_layers=3, token_mixer_hidden_dim=256, channel_mixer_hidden_dim=512)

    def forward(self, x):
        quarter_length = 2

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]
        x17 = x[:, 16:16 + quarter_length, :]
        x18 = x[:, 17:17 + quarter_length, :]
        x19 = x[:, 18:18 + quarter_length, :]
        # 分别传入mix1网络
        out1 = self.mix1(x1) # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        out17 = self.mix1(x17)
        out18 = self.mix1(x18)
        out19 = self.mix1(x19)


        # 组合结果
        combined_out = torch.stack([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16, out17, out18, out19], dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

class himmixs10(nn.Module):
    def __init__(self,input_dim, output_size):
        super(himmixs10, self).__init__()
        self.mix1 =  M_mixs_1(seq_length=10, input_channels=input_dim, num_channels=128, num_layers=2, token_mixer_hidden_dim=128, channel_mixer_hidden_dim=128)
        self.mix2 = M_mixs_2(seq_length=3, input_channels=128, num_classes=output_size, num_channels=128, num_layers=3, token_mixer_hidden_dim=256, channel_mixer_hidden_dim=512)

    def forward(self, x):
        quarter_length = 10

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 5:5 + quarter_length, :]
        x3 = x[:, 10:10 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1) # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        # 组合结果
        combined_out = torch.stack([out1, out2, out3], dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

#----------------------------------------------------------------------------------------------------------

# 深度神经网络DNN模型 类
class TimeSeriesDNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(TimeSeriesDNN, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # 由于输入是三维时间序列，需要将其展平为二维张量
        x = x.view(x.size(0), -1)
        return self.model(x)

#----------------------------------------------------------------------------------------------------------

# 深度卷积神经网络CNN 类
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels, conv_channels, fc_sizes, num_classes):
        super(TimeSeriesCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels

        self.fc_input_size = self._calculate_fc_input_size(input_channels, conv_channels)
        self.fc_layers = nn.ModuleList()
        in_size = self.fc_input_size
        for fc_size in fc_sizes:
            self.fc_layers.append(nn.Linear(in_size, fc_size))
            self.fc_layers.append(nn.ReLU())
            in_size = fc_size
        self.fc_layers.append(nn.Linear(in_size, num_classes))

    def _calculate_fc_input_size(self, input_channels, conv_channels):
        x = torch.randn(1, input_channels, 20) #输入序列的长度
        for layer in self.conv_layers:
            x = layer(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        return x

#----------------------------------------------------------------------------------------------------------

class CNNLSTMSA(nn.Module):
    """
    CNN -> LSTM -> Self-Attention -> (mean pool) -> FC
    输入:  x [B, T, input_dim]
    输出:  logits [B, num_classes]
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels=(64, 128),
        lstm_hidden=256,
        lstm_layers=3,
        bidirectional=False,
        attention_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        # ---- CNN(1D) 特征提取：卷积若干层，但只池化一次（对齐原文） ----
        cnn = []
        in_ch = input_dim
        for out_ch in conv_channels:
            cnn += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        # 只在最后加一次 pooling（window=2）
        cnn += [nn.MaxPool1d(kernel_size=2)]

        self.cnn = nn.Sequential(*cnn)

        # ---- LSTM：输入特征维 = CNN 输出通道数 ----
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---- Self-Attention：对 LSTM 输出序列做时序注意力 ----
        self.attn = SelfAttention(lstm_out_dim, attention_dim)

        # ---- 分类头 ----
        self.head = nn.Sequential(
            nn.Linear(attention_dim, num_classes)
        )

    def forward(self, x, return_attn=False):
        """
        x: [B, T, input_dim]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, T, input_dim], got {tuple(x.shape)}")

        # CNN 需要 [B, C, T]
        x = x.permute(0, 2, 1)          # [B, input_dim, T]
        x = self.cnn(x)                 # [B, Cc, T']

        # LSTM 需要 [B, T', Cc]
        x = x.permute(0, 2, 1)          # [B, T', Cc]
        lstm_out, _ = self.lstm(x)      # [B, T', H(*dir)]

        attn_out, attn_w = self.attn(lstm_out)  # attn_out: [B, T', A], attn_w: [B, T', T']

        # 时间维池化（你也可以换成 max 或者取最后时刻）
        pooled = attn_out.mean(dim=1)   # [B, A]
        logits = self.head(pooled)      # [B, num_classes]

        if return_attn:
            return logits, attn_w
        return logits

#----------------------------------------------------------------------------------------------------------

# 循环神经网络RNN 类
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义 RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层，用于将 RNN 输出映射到类别数
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 RNN
        out, _ = self.rnn(x, h0)

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 通过全连接层得到最终输出
        out = self.fc(out)
        return out

# BIGAT 类
class BIGAT(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BIGAT, self).__init__()
        self.BIGRUSA = BiGRUsa(input_size, 128, 256, num_layers=2)
        # 定义Transformer
        self.tr = TransformerClassifier(input_dim=256, d_model=256, nhead=8, num_encoder_layers=2,
                     dim_feedforward=512, dropout=0, num_classes=num_classes)

    def forward(self, x):
        attention_out = self.BIGRUSA(x)  # GRU 输出，形状: [batch_size, seq_len, hidden_dim*2]
        # 通过全连接层得到最终输出
        out = self.tr(attention_out)
        return out

class TransformerClassifier1(nn.Module):
    def __init__(self, input_dim=3, d_model=1024, nhead=8, num_encoder_layers=6,
                 dim_feedforward=4096, dropout=0.1, num_classes=15):
        super(TransformerClassifier1, self).__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model) #JIA
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
            ),
            num_layers=num_encoder_layers
        )
        self.pool = AttnPool(d_model)
        self.fc_out = nn.Linear(d_model, num_classes)  #线性层将 d_model 的特征投影到 num_classes 维度。

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.input_fc(src) * math.sqrt(self.d_model)
        src = self.norm(src)  #JIA
        src = self.pos_encoder(src)

        # 确保 Transformer 兼容 PyTorch 2.x 的 `forward()` 调用方式
        if src_mask is not None:
            output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.pool(output)  # Global average pooling over the sequence length
        return self.fc_out(output)

class BIGAT1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BIGAT1, self).__init__()
        self.BIGRUSA = BiGRUsa(input_size, 128, 256, num_layers=1)
        # 定义Transformer
        self.tr = TransformerClassifier1(input_dim=256, d_model=256, nhead=8, num_encoder_layers=1,
                     dim_feedforward=512, dropout=0, num_classes=num_classes)

    def forward(self, x):
        attention_out = self.BIGRUSA(x)  # GRU 输出，形状: [batch_size, seq_len, hidden_dim*2]
        # 通过全连接层得到最终输出
        out = self.tr(attention_out)
        return out
#----------------------------------------------------------------------------------------------------------

class TinyTransformerBlock(nn.Module):
    """
    轻量 Transformer Block:
    - Multi-Head Self-Attention (全局)
    - FFN
    - 残差 + LayerNorm
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,   # 方便直接用 [B, T, C]
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        # --- Self-Attention ---
        attn_out, _ = self.self_attn(x, x, x)    # [B, T, C]
        x = self.norm1(x + self.dropout1(attn_out))

        # --- FFN ---
        ffn_out = self.linear2(self.dropout2(F.relu(self.linear1(x))))  # [B, T, C]
        x = self.norm2(x + ffn_out)
        return x

class CNN_TinyTransformer_Classifier(nn.Module):
    """
    CNN + 1~2 层 Tiny Transformer（全局 MHSA）+ CLS池化 的时序分类模型

    Input:  x [B, T, Din]    (比如 T=20, Din=3)
    Output: logits [B, num_classes]
    """
    def __init__(
        self,
        input_dim: int = 3,
        num_classes: int = 15,
        channels: int = 128,
        num_conv_layers: int = 2,
        kernel_size: int = 5,
        seq_len: int = 20,          # 你这边 T=20，可以改成更大
        num_transformer_layers: int = 1,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()

        assert num_conv_layers >= 1
        padding = kernel_size // 2

        # -------- CNN Encoder --------
        convs = []
        in_ch = input_dim
        for i in range(num_conv_layers):
            convs += [
                nn.Conv1d(in_ch, channels, kernel_size=kernel_size, padding=padding, bias=True),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                convs.append(nn.Dropout(dropout))
            in_ch = channels

        self.conv = nn.Sequential(*convs)

        # 用在 [B, T, C] 上的 LayerNorm（可选）
        self.use_layernorm = use_layernorm
        self.ln = nn.LayerNorm(channels) if use_layernorm else nn.Identity()

        # -------- Tiny Transformer Encoder --------
        # 可学习的 CLS token 和 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, channels))  # 每个时间步一个位置向量

        self.transformer_layers = nn.ModuleList([
            TinyTransformerBlock(
                d_model=channels,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_transformer_layers)
        ])

        # -------- Classifier Head（用 CLS 向量） --------
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        # x: [B, T, Din]
        B, T, _ = x.shape

        # ---- CNN encoder ----
        # [B, T, Din] -> [B, Din, T]
        x = x.transpose(1, 2)
        # CNN: [B, C, T]
        x = self.conv(x)
        # 回到 [B, T, C]
        x = x.transpose(1, 2)  # [B, T, C]

        x = self.ln(x)  # 可选 LN

        # ---- 加位置编码 ----
        # 如果实际 T <= seq_len，就截取前 T 个位置向量
        pos = self.pos_embed[:, :T, :]  # [1, T, C]
        x = x + pos

        # ---- 拼 CLS token ----
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat([cls_token, x], dim=1)          # [B, 1+T, C]

        # ---- 通过 Tiny Transformer 层 ----
        for layer in self.transformer_layers:
            x = layer(x)                              # [B, 1+T, C]

        # ---- CLS 池化 ----
        cls_out = x[:, 0, :]                          # [B, C]

        # ---- 分类头 ----
        logits = self.classifier(cls_out)             # [B, num_classes]
        return logits


#----------------------------------------------------------------------------------------------------------
# 分层模型 类 (lstm+lstm)
class CombinedModel1(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel1, self).__init__()
        self.mix1 = LSTMNetwork1(input_dim, 256, 2, 128)
        self.mix2 = LSTMNetwork(128, 256, 4, output_size)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类（lstm+cnnlstmsa）
class CombinedModel2(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel2, self).__init__()
        self.mix1 = LSTMNetwork1(input_dim, 256, 2, 128)
        self.mix2 = CNNLSTMSA(input_dim=128,num_classes = output_size ,conv_channels=(64,128),
         lstm_hidden=256,lstm_layers=4,bidirectional=False,attention_dim=256,
         dropout=0)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类（lstm+bigrusa）
class CombinedModel3(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel3, self).__init__()
        self.mix1 = LSTMNetwork1(input_dim, 256, 2, 128)
        self.mix2 = BiGRUWithSelfAttention(128, hidden_dim=512, attention_dim=160, num_classes=output_size,
                                   num_layers=4)  # BIGRU-SA

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类（lstm+bigat）
class CombinedModel4(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel4, self).__init__()
        self.mix1 = LSTMNetwork1(input_dim, 256, 2, 128)
        self.mix2 = BIGAT(128, output_size)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类（lstm+m-mixs）
class CombinedModel5(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel5, self).__init__()
        self.mix1 = LSTMNetwork1(input_dim, 256, 2, 128)
        self.mix2 = M_mixs_2(seq_length=16, input_channels=128, num_classes=output_size, num_channels=128, num_layers=3, token_mixer_hidden_dim=256, channel_mixer_hidden_dim=512)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类 (lstm+lstm)
class CombinedModel6(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel6, self).__init__()
        self.mix1 = CNNLSTMSA1(input_dim=3,num_classes = 128 ,conv_channels=(128,),
         lstm_hidden=256,lstm_layers=2,bidirectional=False,attention_dim=256,
         dropout=0)
        self.mix2 = LSTMNetwork(128, 256, 4, output_size)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类（lstm+cnnlstmsa）
class CombinedModel7(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel7, self).__init__()
        self.mix1 = CNNLSTMSA1(input_dim=3,num_classes = 128 ,conv_channels=(128,),
         lstm_hidden=256,lstm_layers=2,bidirectional=False,attention_dim=256,
         dropout=0)
        self.mix2 = CNNLSTMSA(input_dim=128,num_classes = output_size ,conv_channels=(64,128),
         lstm_hidden=256,lstm_layers=4,bidirectional=False,attention_dim=256,
         dropout=0)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类（lstm+bigrusa）
class CombinedModel8(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel8, self).__init__()
        self.mix1 =  CNNLSTMSA1(input_dim=3,num_classes = 128 ,conv_channels=(128,),
         lstm_hidden=256,lstm_layers=2,bidirectional=False,attention_dim=256,
         dropout=0)
        self.mix2 = BiGRUWithSelfAttention(128, hidden_dim=512, attention_dim=160, num_classes=output_size,
                                   num_layers=4)  # BIGRU-SA

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类（lstm+bigat）
class CombinedModel9(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel9, self).__init__()
        self.mix1 = CNNLSTMSA1(input_dim=3,num_classes = 128 ,conv_channels=(128,),
         lstm_hidden=256,lstm_layers=2,bidirectional=False,attention_dim=256,
         dropout=0)
        self.mix2 = BIGAT(128, output_size)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类（lstm+m-mixs）
class CombinedModel10(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel10, self).__init__()
        self.mix1 =CNNLSTMSA1(input_dim=3,num_classes = 128 ,conv_channels=(128,),
         lstm_hidden=256,lstm_layers=2,bidirectional=False,attention_dim=256,
         dropout=0)
        self.mix2 = M_mixs_2(seq_length=16, input_channels=128, num_classes=output_size, num_channels=128, num_layers=3, token_mixer_hidden_dim=256, channel_mixer_hidden_dim=512)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out



# 分层模型 类 (lstm+lstm)
class CombinedModel11(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel11, self).__init__()
        self.mix1 = BiGRUWithSelfAttention1(3, hidden_dim=256, attention_dim=160, num_classes=128,
                                   num_layers=2)
        self.mix2 = LSTMNetwork(128, 256, 4, output_size)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类（lstm+cnnlstmsa）
class CombinedModel12(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel12, self).__init__()
        self.mix1 = BiGRUWithSelfAttention1(3, hidden_dim=512, attention_dim=160, num_classes=128,
                                   num_layers=2)
        self.mix2 = CNNLSTMSA(input_dim=128,num_classes = output_size ,conv_channels=(64,128),
         lstm_hidden=256,lstm_layers=4,bidirectional=False,attention_dim=256,
         dropout=0)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类（lstm+bigrusa）
class CombinedModel13(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel13, self).__init__()
        self.mix1 = BiGRUWithSelfAttention1(3, hidden_dim=512, attention_dim=160, num_classes=128,
                                   num_layers=2)
        self.mix2 = BiGRUWithSelfAttention(128, hidden_dim=512, attention_dim=160, num_classes=output_size,
                                   num_layers=4)  # BIGRU-SA

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类（lstm+bigat）
class CombinedModel14(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel14, self).__init__()
        self.mix1 = BiGRUWithSelfAttention1(3, hidden_dim=512, attention_dim=160, num_classes=128,
                                   num_layers=2)
        self.mix2 = BIGAT(128, output_size)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类（lstm+m-mixs）
class CombinedModel15(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel15, self).__init__()
        self.mix1 = BiGRUWithSelfAttention1(3, hidden_dim=512, attention_dim=160, num_classes=128,
                                   num_layers=2)
        self.mix2 = M_mixs_2(seq_length=16, input_channels=128, num_classes=output_size, num_channels=128, num_layers=3, token_mixer_hidden_dim=256, channel_mixer_hidden_dim=512)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类 (lstm+lstm)
class CombinedModel16(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel16, self).__init__()
        self.mix1 = BIGAT1(input_dim, 128)
        self.mix2 = LSTMNetwork(128, 256, 4, output_size)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类（lstm+cnnlstmsa）
class CombinedModel17(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel17, self).__init__()
        self.mix1 =BIGAT1(input_dim, 128)
        self.mix2 = CNNLSTMSA(input_dim=128,num_classes = output_size ,conv_channels=(64,128),
         lstm_hidden=256,lstm_layers=4,bidirectional=False,attention_dim=256,
         dropout=0)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类（lstm+bigrusa）
class CombinedModel18(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel18, self).__init__()
        self.mix1 = BIGAT1(input_dim, 128)
        self.mix2 = BiGRUWithSelfAttention(128, hidden_dim=512, attention_dim=160, num_classes=output_size,
                                   num_layers=4)  # BIGRU-SA

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类（lstm+bigat）
class CombinedModel19(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel19, self).__init__()
        self.mix1 = BIGAT1(input_dim, 128)
        self.mix2 = BIGAT(128, output_size)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类（lstm+m-mixs）
class CombinedModel20(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel20, self).__init__()
        self.mix1 = BIGAT1(input_dim, 128)
        self.mix2 = M_mixs_2(seq_length=16, input_channels=128, num_classes=output_size, num_channels=128, num_layers=3, token_mixer_hidden_dim=256, channel_mixer_hidden_dim=512)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类 (lstm+lstm)
class CombinedModel21(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel21, self).__init__()
        self.mix1 = M_mixs_1(seq_length=5, input_channels=input_dim, num_channels=128, num_layers=2, token_mixer_hidden_dim=128, channel_mixer_hidden_dim=128)
        self.mix2 = LSTMNetwork(128, 256, 4, output_size)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类（lstm+cnnlstmsa）
class CombinedModel22(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel22, self).__init__()
        self.mix1 =M_mixs_1(seq_length=5, input_channels=input_dim, num_channels=128, num_layers=2, token_mixer_hidden_dim=128, channel_mixer_hidden_dim=128)
        self.mix2 = CNNLSTMSA(input_dim=128,num_classes = output_size ,conv_channels=(64,128),
         lstm_hidden=256,lstm_layers=4,bidirectional=False,attention_dim=256,
         dropout=0)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out

# 分层模型 类（lstm+bigrusa）
class CombinedModel23(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel23, self).__init__()
        self.mix1 = M_mixs_1(seq_length=5, input_channels=input_dim, num_channels=128, num_layers=2, token_mixer_hidden_dim=128, channel_mixer_hidden_dim=128)
        self.mix2 = BiGRUWithSelfAttention(128, hidden_dim=512, attention_dim=160, num_classes=output_size,
                                   num_layers=4)  # BIGRU-SA

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


# 分层模型 类（lstm+bigat）
class CombinedModel24(nn.Module):
    def __init__(self, input_dim, output_size):
        super(CombinedModel24, self).__init__()
        self.mix1 = M_mixs_1(seq_length=5, input_channels=input_dim, num_channels=128, num_layers=2, token_mixer_hidden_dim=128, channel_mixer_hidden_dim=128)
        self.mix2 = BIGAT(128, output_size)

    def forward(self, x):
        quarter_length = x.size(1) // 4

        # 划分成四份
        x1 = x[:, :quarter_length, :]
        x2 = x[:, 1:1 + quarter_length, :]
        x3 = x[:, 2:2 + quarter_length, :]
        x4 = x[:, 3:3 + quarter_length, :]
        x5 = x[:, 4:4 + quarter_length, :]
        x6 = x[:, 5:5 + quarter_length, :]
        x7 = x[:, 6:6 + quarter_length, :]
        x8 = x[:, 7:7 + quarter_length, :]
        x9 = x[:, 8:8 + quarter_length, :]
        x10 = x[:, 9:9 + quarter_length, :]
        x11 = x[:, 10:10 + quarter_length, :]
        x12 = x[:, 11:11 + quarter_length, :]
        x13 = x[:, 12:12 + quarter_length, :]
        x14 = x[:, 13:13 + quarter_length, :]
        x15 = x[:, 14:14 + quarter_length, :]
        x16 = x[:, 15:15 + quarter_length, :]

        # 分别传入mix1网络
        out1 = self.mix1(x1)  # b*128
        out2 = self.mix1(x2)
        out3 = self.mix1(x3)
        out4 = self.mix1(x4)
        out5 = self.mix1(x5)
        out6 = self.mix1(x6)
        out7 = self.mix1(x7)
        out8 = self.mix1(x8)
        out9 = self.mix1(x9)
        out10 = self.mix1(x10)
        out11 = self.mix1(x11)
        out12 = self.mix1(x12)
        out13 = self.mix1(x13)
        out14 = self.mix1(x14)
        out15 = self.mix1(x15)
        out16 = self.mix1(x16)
        # 组合结果
        combined_out = torch.stack(
            [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16],
            dim=1)  # b*6*128

        # 传入 mix2
        final_out = self.mix2(combined_out)
        return final_out


