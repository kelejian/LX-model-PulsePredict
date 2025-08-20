import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from torch_geometric.nn import MLP as PygMLP # 直接用PyG的MLP模块

class PulseMLP(BaseModel):
    """
    基于多层感知机(MLP)的模型，用于从工况参数预测碰撞波形。

    此模型输出高斯分布的均值和方差，以配合 GaussianNLLLoss 使用。
    """
    def __init__(self, input_dim=3, output_channels=3, output_points=200, 
                 hidden_dims=256, num_layers=3, dropout=0.2):
        """
        模型初始化。
        
        :param input_dim: 输入特征的维度 (工况参数数量, 默认为3)。
        :param output_channels: 输出波形的通道数 (x, y, z三轴, 默认为3)。
        :param output_points: 每条波形的时间点数 (默认为200)。
        :param hidden_dims: MLP 隐藏层的维度。
        :param num_layers: MLP 的层数。
        :param dropout: Dropout 概率。
        """
        super().__init__()
        
        self.output_channels = output_channels
        self.output_points = output_points
        output_dim = output_channels * output_points # 3 * 200 = 600

        # 1. 定义一个共享的MLP骨干网络，用于从低维输入中提取高维特征
        self.backbone = PygMLP(
            in_channels=input_dim,
            hidden_channels=hidden_dims,
            out_channels=hidden_dims,    # 骨干网络的输出是高维特征向量
            num_layers=num_layers,
            norm="batch_norm",           # 使用批归一化
            act="leaky_relu",            # 使用 LeakyReLU 激活函数
            plain_last=False,            # 对骨干网络的最后一层也应用激活和归一化
            dropout=dropout
        )
        
        # 2. 定义两个独立的"头"（线性层），分别用于预测均值和对数方差
        # 均值头
        self.mean_head = nn.Linear(hidden_dims + input_dim, output_dim)
        
        # 对数方差头 (预测log(var)而不是直接预测var，以保证方差为正，且训练更稳定)
        self.log_var_head = nn.Linear(hidden_dims + input_dim, output_dim)

    def forward(self, x):
        """
        前向传播函数。

        :param x: 输入张量，形状为 (batch_size, 3)。
        :return: 一个元组，包含均值和方差张量 (mean, variance)，
                 每个张量的形状都为 (batch_size, 3, 200)。
        """
        # x 形状: (batch_size, 3)
        
        # 通过骨干网络提取特征
        latent_features = self.backbone(x)  # -> (batch_size, hidden_dims)

        # concat特征和输入
        latent_features = torch.cat([latent_features, x], dim=-1)  # -> (batch_size, hidden_dims + 3)

        # 从特征中预测均值
        mean = self.mean_head(latent_features) # -> (batch_size, 600)
        
        # 从特征中预测对数方差
        log_var = self.log_var_head(latent_features) # -> (batch_size, 600)
        
        # 将均值和对数方差重塑为波形的目标形状
        mean = mean.view(-1, self.output_channels, self.output_points) # -> (batch_size, 3, 200)
        log_var = log_var.view(-1, self.output_channels, self.output_points) # -> (batch_size, 3, 200)

        # 通过指数运算得到方差，确保其为正值
        # 加上一个小的epsilon可以增加数值稳定性，防止方差为0
        variance = torch.exp(log_var)
        
        return mean, variance
