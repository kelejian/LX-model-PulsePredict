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
    def __init__(self, input_dim=3, output_channels=3, output_points=150, 
                 hidden_dims=256, num_layers=3, dropout=0.2, GauNll_use=True):
        """
        模型初始化。
        
        :param input_dim: 输入特征的维度 (工况参数数量, 默认为3)。
        :param output_channels: 输出波形的通道数 (x, y, z三轴, 默认为3)。
        :param output_points: 每条波形的时间点数 (默认为150)。
        :param hidden_dims: MLP 隐藏层的维度。
        :param num_layers: MLP 的层数。
        :param dropout: Dropout 概率。
        :param GauNll_use: 是否使用高斯NLLLoss，若为False，则只输出均值。
        """
        super().__init__()
        
        self.output_channels = output_channels
        self.output_points = output_points
        self.GauNll_use = GauNll_use
        output_dim = output_channels * output_points # 3 * 150 = 450

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
        # 2. 定义预测头
        self.mean_head = nn.Linear(hidden_dims + input_dim, output_dim)
        if self.GauNll_use:
            # 对数方差头 (预测log(var)而不是直接预测var，以保证方差为正，且训练更稳定)
            self.log_var_head = nn.Linear(hidden_dims + input_dim, output_dim)

    def forward(self, x):
        """
        前向传播函数。
        """
        # x 形状: (batch_size, 3)
        latent_features = self.backbone(x)  # -> (batch_size, hidden_dims)
        latent_features = torch.cat([latent_features, x], dim=-1)  # -> (batch_size, hidden_dims + 3)

        mean = self.mean_head(latent_features) # -> (batch_size, 450)
        mean = mean.view(-1, self.output_channels, self.output_points) # -> (batch_size, 3, 150)

        if not self.GauNll_use:
            return mean

        log_var = self.log_var_head(latent_features) # -> (batch_size, 450)
        log_var = log_var.view(-1, self.output_channels, self.output_points) # -> (batch_size, 3, 150)
        variance = torch.exp(log_var)
        
        return mean, variance

    def compute_loss(self, model_output, target, criterions):
            total_loss = torch.tensor(0.0, device=target.device)
            loss_components = {}
            for criterion_item in criterions:
                loss_instance = criterion_item['instance']
                weight = criterion_item['weight']
                loss_type_name = type(loss_instance).__name__
                # For MLP, the logic is simpler as it doesn't have multi-scale outputs
                if self.GauNll_use:
                    pred_mean, pred_var = model_output
                    loss = loss_instance(pred_mean, pred_var, target)
                else:
                    loss = loss_instance(model_output, target)
                
                total_loss += weight * loss
                loss_components[loss_type_name] = loss.item()

            return total_loss, loss_components

    def get_metrics_output(self, model_output):
        if self.GauNll_use:
            # The first element of the tuple is the mean prediction
            return model_output[0]
        else:
            # The output is the prediction itself
            return model_output

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
    def __init__(self, in_channels, out_channels, UpLayer_ks=3, ResLayer_ks=3, target_length=None):
        super().__init__()
        self.target_length = target_length
        # 步骤1: 定义一个标准卷积，用于对插值后的特征进行可学习的变换
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=UpLayer_ks, padding='same')
        # 步骤2: 使用残差块进行深度特征精调
        self.res_block = ResBlock1D(out_channels, out_channels, ResLayer_ks)

    def forward(self, x):
        """
        前向传播。
        :param x: 输入特征序列, 形状: (B, C_in, L_in)
        :return: 输出特征序列, 形状: (B, C_out, target_length)
        """
        # 动作1: 使用非学习性的线性插值上采样到目标长度
        x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        # 动作2: 通过卷积层进行特征变换
        x = self.conv(x)
        # 动作3: 通过残差块进行精调
        x = self.res_block(x)
        return x

class PulseCNN_V1(BaseModel):
    """
    基于1D-CNN的碰撞波形预测模型。

    该模型采用层级式多尺度生成架构，其上采样单元为残差精调模块，
    旨在实现从粗到精的“逐步细化预测”。
    """
    def __init__(self, input_dim=3, output_dim=150, output_channels=3, 
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
class PulseCNN(BaseModel):
    """
    基于1D-CNN的碰撞波形预测模型 (V2 - 参数高效版)。
    该模型采用层级式多尺度生成架构，其上采样单元为残差精调模块，
    旨在实现从粗到精的“逐步细化预测”。
    """
    def __init__(self, input_dim=3, output_channels=3,
                 mlp_latent_dim=256, mlp_num_layers=3,
                 projection_init_channels=16, UpLayer_ks=3, ResLayer_ks=3,
                 channels_list=[256, 128, 64], output_lengths=None, GauNll_use=True):
        super().__init__()
        
        # --- 输入参数校验 ---
        if output_lengths is None:
            raise ValueError("`output_lengths` must be provided.")
        if len(channels_list) != len(output_lengths):
            raise ValueError(f"Length of `channels_list` ({len(channels_list)}) must match length of `output_lengths` ({len(output_lengths)}).")

        self.initial_length = output_lengths[0]
        self.output_dim = output_lengths[-1]
        self.GauNll_use = GauNll_use

        # 1. 输入编码器
        self.encoder = PygMLP(
            in_channels=input_dim, hidden_channels=mlp_latent_dim, out_channels=mlp_latent_dim,
            num_layers=mlp_num_layers, norm="batch_norm", act="leaky_relu", plain_last=False, dropout=0.1
        )

        # 2. 投影块
        self.projection_block = nn.Sequential(
            nn.Linear(mlp_latent_dim + input_dim, self.initial_length * projection_init_channels),
            nn.BatchNorm1d(self.initial_length * projection_init_channels), nn.LeakyReLU(),
            nn.Conv1d(projection_init_channels, channels_list[0], kernel_size=1)
        )
        self.projection_init_channels = projection_init_channels
        self.initial_shape_after_proj = (channels_list[0], self.initial_length)

        # 3. 多尺度解码器
        self.decoder_blocks = nn.ModuleList()
        num_upsamples = len(channels_list) - 1
        for i in range(num_upsamples):
            self.decoder_blocks.append(
                UpsamplingBlock(in_channels=channels_list[i], 
                              out_channels=channels_list[i+1], 
                              UpLayer_ks=UpLayer_ks, 
                              ResLayer_ks=ResLayer_ks, 
                              target_length=output_lengths[i+1]) # 传递下一阶段的目标长度
            )

        # 4. 多尺度输出头
        self.output_heads = nn.ModuleList()
        for channels in channels_list:
            head_modules = {'mean': nn.Conv1d(channels, output_channels, kernel_size=1)}
            if self.GauNll_use:
                head_modules['log_var'] = nn.Conv1d(channels, output_channels, kernel_size=1)
            self.output_heads.append(nn.ModuleDict(head_modules))

    def forward(self, x):
        # 1. 编码与信息融合
        z = self.encoder(x)
        z_prime = torch.cat([z, x], dim=-1)

        # 2. 通过投影块生成初始序列
        proj_out = self.projection_block[0](z_prime)
        proj_out = self.projection_block[1](proj_out)
        proj_out = self.projection_block[2](proj_out)
        proj_out_reshaped = proj_out.view(-1, self.projection_init_channels, self.initial_length)
        initial_sequence = self.projection_block[3](proj_out_reshaped)

        # 3. 解码与多尺度特征提取
        features_list = [initial_sequence]
        z_decoded = initial_sequence
        for block in self.decoder_blocks:
            z_decoded = block(z_decoded)
            features_list.append(z_decoded)

        # 4. 多尺度预测
        mean_preds = []
        log_var_preds = [] if self.GauNll_use else None

        for features, head in zip(features_list, self.output_heads):
            mean = head['mean'](features)
            mean_preds.append(mean)
            if self.GauNll_use:
                log_var = head['log_var'](features)
                log_var_preds.append(log_var)

        if not self.GauNll_use:
            return mean_preds
        else:
            var_preds = [torch.exp(log_var) for log_var in log_var_preds]
            return mean_preds, var_preds

    def compute_loss(self, model_output, target, criterions):
            """
            准备多尺度预测和目标，并调用criterions计算加权总损失。
            返回总损失和一个包含各分量损失的字典。
            """
            total_loss = torch.tensor(0.0, device=target.device)
            loss_components = {}
            # --- 准备通用的输入 ---
            if self.GauNll_use:
                pred_mean_list, pred_var_list = model_output
                pred_list_for_multiloss = list(zip(pred_mean_list, pred_var_list))
            else:
                pred_mean_list = model_output
                pred_list_for_multiloss = pred_mean_list
            
            # 为每个尺度的预测准备对应的目标
            target_list = [F.interpolate(target, size=pred_mean.shape[-1], mode='linear', align_corners=False) for pred_mean in pred_mean_list]
            target_list[-1] = target # 最后为原始尺度的真值
            # 如果最后一个尺度的长度与真值长度不符，则报错
            if target_list[-1].shape[-1] != self.output_dim:
                raise ValueError(f"最后一个尺度的长度 {target_list[-1].shape[-1]} 与真值长度 {self.output_dim} 不符")

            # --- 遍历所有损失函数并计算加权和 ---
            for criterion_item in criterions:
                loss_instance = criterion_item['instance']
                weight = criterion_item['weight']
                loss_type_name = type(loss_instance).__name__

                # --- 核心逻辑: 根据损失类型，传递不同的参数 ---
                if loss_type_name == 'MultiLoss':
                    loss = loss_instance(pred_list_for_multiloss, target_list)
                # 示例：未来可在此处为其他loss添加elif分支
                # elif loss_type_name == 'YourFuturePhysicsLoss':
                #     loss = loss_instance(...)
                else:
                    # 提供一个默认行为给简单的、非多尺度的损失函数
                    final_pred = self.get_metrics_output(model_output)
                    loss = loss_instance(final_pred, target)

                total_loss += weight * loss
                loss_components[loss_type_name] = loss.item()

            return total_loss, loss_components

    def get_metrics_output(self, model_output):
        if self.GauNll_use:
            # model_output is (pred_mean_list, pred_var_list)
            return model_output[0][-1]  # Return the last mean prediction
        else:
            # model_output is pred_mean_list
            return model_output[-1]
