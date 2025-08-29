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


#==========================================================================================
# PulseCNN (层级式多尺度生成 + 残差精调)
#==========================================================================================
class ResBlock1D(BaseModel):
    """
    一维残差块 (1D Residual Block)。
    对特征序列进行深度非线性变换，是特征精调的核心单元。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # 保证卷积操作后序列长度不变
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 捷径连接：如果输入输出通道数不同，使用1x1卷积进行维度匹配
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        """
        前向传播。
        输入 x -> [Conv1 -> BN1 -> ReLU] -> [Conv2 -> BN2] -> + -> ReLU -> 输出
                                                              |
                                                          [Shortcut]
        :param x: 输入特征序列, 形状: (B, C_in, L)
        :return: 输出特征序列, 形状: (B, C_out, L)
        """
        residual = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.leaky_relu(out)
class UpsamplingBlock(BaseModel):
    """
    一维残差上采样块 (1D Residual Upsampling Block)。
    采用 'Upsample + Conv1d' 策略替代转置卷积，以避免棋盘效应。
    """
    def __init__(self, in_channels, out_channels, UpLayer_ks=1, scale_factor=2):
        super().__init__()
        # 结合了“尺度提升”和“特征变换”的模块
        self.upsample_conv = nn.Sequential(
            # 步骤1: 使用非学习性的线性插值进行上采样，保证平滑无伪影
            nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False),
            # 步骤2: 使用标准卷积对插值后的特征进行可学习的变换
            nn.Conv1d(in_channels, out_channels, kernel_size=UpLayer_ks, padding='same'),
        )
        # 步骤3: 使用残差块进行深度特征精调
        self.res_block = ResBlock1D(out_channels, out_channels)

    def forward(self, x):
        """
        前向传播。
        :param x: 输入特征序列, 形状: (B, C_in, L)
        :return: 输出特征序列, 形状: (B, C_out, L * scale_factor)
        """
        # x 形状: (B, C_in, L)
        x = self.upsample_conv(x)  # -> (B, C_out, L * scale_factor)
        x = self.res_block(x)      # -> (B, C_out, L * scale_factor)
        return x


class PulseCNN_V1(BaseModel):
    """
    基于1D-CNN的碰撞波形预测模型 (A+B综合方案)。

    该模型采用层级式多尺度生成架构，其上采样单元为残差精调模块，
    旨在实现从粗到精的“逐步细化预测”。
    """
    def __init__(self, input_dim=3, output_dim=200, output_channels=3, 
                 mlp_latent_dim=256, mlp_num_layers=2, 
                 channels_list=[256, 128, 64, 32], scale_factor=2 ):
        super().__init__()
        
        # 动态计算解码器所需的初始序列长度
        num_upsamples = len(channels_list) - 1
        self.initial_length = output_dim // (scale_factor ** num_upsamples)
        self.output_dim = output_dim

        # 1. 输入编码器: 将工况参数映射为潜在向量
        self.encoder = PygMLP(
            in_channels=input_dim,
            hidden_channels=mlp_latent_dim,
            out_channels=channels_list[0] * self.initial_length,
            num_layers=mlp_num_layers,
            norm="batch_norm",
            act="leaky_relu",
            plain_last=False,  # 对输入编码器的最后一层也应用激活和归一化
            dropout=0.1
        )
        self.initial_shape = (channels_list[0], self.initial_length)

        # 2. 多尺度解码器: 由一系列残差上采样块构成
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_upsamples):
            self.decoder_blocks.append(
                UpsamplingBlock(in_channels=channels_list[i], out_channels=channels_list[i+1], UpLayer_ks=1, scale_factor=scale_factor)
            )

        # 3. 多尺度输出头: 在每个尺度上生成波形预测
        self.output_heads = nn.ModuleList()
        for channels in channels_list:
            head = nn.ModuleDict({
                'mean': nn.Conv1d(channels, output_channels, kernel_size=1),
                'log_var': nn.Conv1d(channels, output_channels, kernel_size=1)
            })
            self.output_heads.append(head)

        self.last_length = self.initial_length * (scale_factor ** num_upsamples)

    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量，形状为 (B, 3)。
        :return: 一个元组，包含两个列表 (mean_preds, var_preds)，
                 每个列表都包含了从粗到精的多个尺度的预测。
        """
        # --- 1. 编码与重塑 ---
        # x 形状: (B, 3)
        z = self.encoder(x)  # -> (B, C_0 * L_0)
        z = z.view(-1, *self.initial_shape) # -> (B, C_0, L_0)

        # --- 2. 解码与多尺度特征提取 ---
        features_list = [z]
        for block in self.decoder_blocks:
            z = block(z)
            features_list.append(z)
        # features_list 包含了每个尺度的特征, 形状为:
        # [(B, C_0, L_0), (B, C_1, L_1), ..., (B, C_n, L_n)]

        # --- 3. 多尺度预测 ---
        mean_preds = []
        log_var_preds = []
        for features, head in zip(features_list, self.output_heads):
            # features 形状: (B, C_i, L_i)
            mean = head['mean'](features)      # -> (B, output_channels, L_i)
            log_var = head['log_var'](features)  # -> (B, output_channels, L_i)
            mean_preds.append(mean)
            log_var_preds.append(log_var)
        
        # --- 4. 最终长度校正 ---
        if self.last_length != self.output_dim:
            mean_preds[-1] = F.interpolate(mean_preds[-1], size=self.output_dim, mode='linear', align_corners=False)
            log_var_preds[-1] = F.interpolate(log_var_preds[-1], size=self.output_dim, mode='linear', align_corners=False)

        # --- 5. 计算方差 ---
        var_preds = [torch.exp(log_var) for log_var in log_var_preds]
        
        return mean_preds, var_preds

# ... (ResBlock1D 和 UpsamplingBlock 的定义保持不变, 予以保留) ...

class PulseCNN(BaseModel):
    """
    基于1D-CNN的碰撞波形预测模型 (V2 - 参数高效版)。

    该模型采用您提出的新方案：
    1. 使用一个高效的MLP作为编码器，生成一个低维潜在向量。
    2. 将原始输入与潜在向量拼接，实现信息融合。
    3. 通过一个参数高效的“投影块”（线性层+1x1卷积）将融合后的特征
       映射为解码器所需的初始序列。
    4. 采用层级式多尺度解码器（残差上采样），实现从粗到精的预测。
    """
    def __init__(self, input_dim=3, output_dim=200, output_channels=3,
                 mlp_latent_dim=256, mlp_num_layers=3,
                 projection_init_channels=16,
                 channels_list=[256, 128, 64, 32], scale_factor=2):
        """
        :param input_dim: 输入工况参数的维度。
        :param output_dim: 最终输出波形的时间点数。
        :param output_channels: 输出波形的通道数 (x, y, z三轴)。
        :param mlp_latent_dim: MLP编码器输出的潜在向量维度。
        :param mlp_num_layers: MLP编码器的层数。
        :param projection_init_channels: 投影块中间层的初始通道数。
        :param channels_list: 解码器中每个层级的通道数列表。
        :param scale_factor: 上采样因子。
        """
        super().__init__()
        
        num_upsamples = len(channels_list) - 1
        self.initial_length = output_dim // (scale_factor ** num_upsamples)
        self.output_dim = output_dim

        # 1. 输入编码器: 将工况参数高效地映射为低维潜在向量 z
        self.encoder = PygMLP(
            in_channels=input_dim,
            hidden_channels=mlp_latent_dim,
            out_channels=mlp_latent_dim, # 输出一个紧凑的潜在向量
            num_layers=mlp_num_layers,
            norm="batch_norm",
            act="leaky_relu",
            plain_last=False, # 对输出层也应用BN和ReLU
            dropout=0.1
        )

        # 2. 投影块 (Projection Block): 高效地将融合特征映射为初始序列
        self.projection_block = nn.Sequential(
            # 步骤1: 小型线性层，进行低维时序投影
            nn.Linear(mlp_latent_dim + input_dim, self.initial_length * projection_init_channels),
            nn.BatchNorm1d(self.initial_length * projection_init_channels),
            nn.LeakyReLU(),
            # (Reshape 操作将在 forward 函数中进行)
            # 步骤2: 高效的1x1卷积，进行通道扩张
            nn.Conv1d(projection_init_channels, channels_list[0], kernel_size=1)
        )
        self.projection_init_channels = projection_init_channels
        self.initial_shape_after_proj = (channels_list[0], self.initial_length)

        # 3. 多尺度解码器: 结构与之前版本保持一致
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_upsamples):
            self.decoder_blocks.append(
                UpsamplingBlock(in_channels=channels_list[i], out_channels=channels_list[i+1], UpLayer_ks=1, scale_factor=scale_factor)
            )

        # 4. 多尺度输出头: 结构与之前版本保持一致
        self.output_heads = nn.ModuleList()
        for channels in channels_list:
            head = nn.ModuleDict({
                'mean': nn.Conv1d(channels, output_channels, kernel_size=1),
                'log_var': nn.Conv1d(channels, output_channels, kernel_size=1)
            })
            self.output_heads.append(head)

        self.last_length = self.initial_length * (scale_factor ** num_upsamples)

    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量，形状为 (B, 3)。
        :return: 一个元组，包含两个列表 (mean_preds, var_preds)。
        """
        # --- 1. 编码与信息融合 ---
        # x 形状: (B, 3)
        z = self.encoder(x)             # -> (B, mlp_latent_dim)
        z_prime = torch.cat([z, x], dim=-1) # -> (B, mlp_latent_dim + 3)

        # --- 2. 通过投影块生成初始序列 ---
        # z_prime 形状: (B, mlp_latent_dim + 3)
        proj_out = self.projection_block[0](z_prime) # Linear: -> (B, L_0 * C_init)
        proj_out = self.projection_block[1](proj_out) # BatchNorm1d
        proj_out = self.projection_block[2](proj_out) # LeakyReLU
        
        # Reshape:
        proj_out_reshaped = proj_out.view(-1, self.projection_init_channels, self.initial_length) # -> (B, C_init, L_0)
        
        # 1x1 Conv for channel expansion:
        initial_sequence = self.projection_block[3](proj_out_reshaped) # -> (B, C_0, L_0)

        # --- 3. 解码与多尺度特征提取 ---
        features_list = [initial_sequence]
        z_decoded = initial_sequence
        for block in self.decoder_blocks:
            z_decoded = block(z_decoded)
            features_list.append(z_decoded)

        # --- 4. 多尺度预测 ---
        mean_preds = []
        log_var_preds = []
        for features, head in zip(features_list, self.output_heads):
            mean = head['mean'](features)
            log_var = head['log_var'](features)
            mean_preds.append(mean)
            log_var_preds.append(log_var)
        
        # --- 5. 最终长度校正 ---
        if self.last_length != self.output_dim:
            mean_preds[-1] = F.interpolate(mean_preds[-1], size=self.output_dim, mode='linear', align_corners=False)
            log_var_preds[-1] = F.interpolate(log_var_preds[-1], size=self.output_dim, mode='linear', align_corners=False)

        # --- 6. 计算方差 ---
        var_preds = [torch.exp(log_var) for log_var in log_var_preds]
        
        return mean_preds, var_preds