import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import model.loss as module_loss # 导入自身模块以便动态创建base_loss

# =========================================================================
# 自动加权loss总管类 (Auto-Weighting Wrapper)
# =========================================================================
class AutoWeightedLoss(nn.Module):
    """
    基于同方差不确定性(Homoscedastic Uncertainty)的多任务自动加权封装器。
    
    原理: 
        L_total = Σ [λ_i * (0.5 * exp(-s_i) * L_i + 0.5 * s_i)]
        其中 s_i = log(σ_i^2) 是可学习参数，λ_i 是人为设定的先验权重。
        该机制能自动平衡不同Loss的数值尺度，同时保留人为设定的任务优先级。
    """
    def __init__(self, loss_configs):
        """
        :param loss_configs: 包含子Loss配置的列表。格式示例:
               [
                   {"type": "MultiScaleLoss", "prior_weight": 1.0, "args": {...}},
                   {"type": "VelocityLoss", "prior_weight": 0.5, "args": {...}}
               ]
        """
        super().__init__()
        self.losses = nn.ModuleList()  # 存储实例化后的子Loss模块
        self.prior_weights = []        # 存储人为设定的λ_i
        self.loss_keys = []            # 存储Loss名称用于日志记录

        # --- 动态构建子Loss ---
        for config in loss_configs:
            loss_type = config['type']
            prior_w = config.get('prior_weight', 1.0) # 默认为1.0
            loss_args = config.get('args', {})

            # 反射实例化: model.loss.LossClass(**args)
            loss_cls = getattr(module_loss, loss_type)
            self.losses.append(loss_cls(**loss_args))
            
            self.prior_weights.append(prior_w)
            self.loss_keys.append(loss_type)

        # --- 初始化可学习参数 s_i ---
        # 初始化为0 (即σ=1)，保证训练初期的梯度稳定性
        self.log_vars = nn.Parameter(torch.zeros(len(self.losses)))
        
        # 将静态权重注册为buffer (不参与梯度更新，但随模型保存)
        self.register_buffer('priors', torch.tensor(self.prior_weights))

    def forward(self, model_output, target):
        """
        :param model_output: 模型输出 (通常是多尺度列表 [s1, s2, s3])
        :param target: 真实标签 (B, C, L)
        :return: (加权总Loss, 各分量Loss字典)
        """
        total_loss = 0
        loss_components = {}

        for i, loss_fn in enumerate(self.losses):
            # 1. 计算原始物理Loss (L_i)
            # 注: 各子Loss内部需自行处理 model_output 格式(如取列表最后一个元素)
            raw_loss = loss_fn(model_output, target)
            
            # 2. 获取参数
            s_i = self.log_vars[i]     # 可学习的不确定性参数
            lambda_i = self.priors[i]  # 固定的人为先验权重

            # 3. 应用自动加权公式 (外乘权重方案)
            # precision = 1/σ^2 = exp(-s)
            precision = torch.exp(-s_i)
            weighted_loss = lambda_i * (0.5 * precision * raw_loss + 0.5 * s_i)
            
            total_loss += weighted_loss
            
            # 4. 记录日志 (记录原始物理Loss值，方便观察实际性能)
            key = self.loss_keys[i]
            loss_components[key] = raw_loss.item() 

        return total_loss, loss_components

# =========================================================================
# 多尺度回归骨干 (Multi-Scale Backbone Loss)
# =========================================================================
class MultiScaleLoss(nn.Module):
    """
    多尺度加权回归损失。
    支持配置基础损失函数(如L1, MSE)，并支持通道加权。
    """
    def __init__(self, scale_weights=[0.1, 0.2, 1.0], base_loss='L1Loss', channel_weights=[1.0, 1.0, 1.0], **kwargs):
        """
        :param scale_weights: 各尺度(s1, s2, s3)的权重列表
        :param base_loss: 基础损失函数类名 (L1Loss, MSELoss, GaussianNLLLoss等)
        :param channel_weights: 各物理通道(x, y, z)的权重
        :param kwargs: 透传给base_loss的参数
        """
        super().__init__()
        self.scale_weights = scale_weights
        self.channel_weights = torch.tensor(channel_weights)
        
        # 动态实例化基础Loss (如 nn.L1Loss)
        # 强制 reduction='none' 以便后续手动应用通道加权
        if base_loss == 'GaussianNLLLoss':
             self.base_criterion = nn.GaussianNLLLoss(reduction='none', **kwargs)
        else:
             loss_cls = getattr(nn, base_loss)
             self.base_criterion = loss_cls(reduction='none', **kwargs)

    def forward(self, preds_list, target):
        """
        :param preds_list: 多尺度预测列表 [s1, s2, s3]
        :param target: 真实标签 (B, C, L)
        """
        # 兼容性处理: 如果模型只返回单尺度，转为列表
        if not isinstance(preds_list, (list, tuple)):
            preds_list = [preds_list]

        total_loss = 0
        device = target.device
        # 调整通道权重形状以支持广播: (C,) -> (1, C, 1)
        c_w = self.channel_weights.to(device).view(1, -1, 1)

        # 遍历每个尺度进行计算
        for i, pred in enumerate(preds_list):
            # 超出配置层数或权重为0则跳过
            if i >= len(self.scale_weights) or self.scale_weights[i] == 0:
                continue
                
            # --- 1. 对齐目标 (Interpolate Target) ---
            # 如果当前尺度预测长度与目标不一致，则缩放目标
            curr_len = pred.shape[-1]
            if curr_len != target.shape[-1]:
                target_resized = F.interpolate(target, size=curr_len, mode='linear', align_corners=False)
            else:
                target_resized = target

            # --- 2. 计算基础Loss ---
            # 区分处理 GaussianNLL (输入为 mean, var) 和 普通回归 (输入为 pred)
            if isinstance(self.base_criterion, nn.GaussianNLLLoss):
                # 假设 GauNLL 模式下 pred 是 (mean, var) 元组
                # 注意：这需要模型输出配合，若模型未开启GauNLL，此分支不应被执行
                mean, var = pred
                loss = self.base_criterion(mean, target_resized, var)
            else:
                loss = self.base_criterion(pred, target_resized)

            # --- 3. 应用通道加权并聚合 ---
            # loss: (B, C, L) * c_w: (1, C, 1) -> Mean
            weighted_loss = (loss * c_w).mean()
            
            total_loss += self.scale_weights[i] * weighted_loss

        return total_loss

class GaussianNLLLoss(nn.Module):
    """
    高斯负对数似然损失函数。

    该损失函数假设目标值(target)服从一个高斯分布（正态分布），
    而模型的任务是预测这个分布的均值(mean)和方差(variance)。
    它适用于回归任务，特别是当需要模型量化其预测的不确定性时。

    Args:
        **kwargs: 传入 torch.nn.GaussianNLLLoss 的额外参数, 
                  例如 full=True, eps=1e-6, reduction='mean'。
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.nll_loss = nn.GaussianNLLLoss(**kwargs)

    def forward(self, pred_mean, pred_var, target):
        """
        前向传播计算损失。
        调整参数顺序以匹配 MultiLoss 的调用。

        :param pred_mean: 模型预测的均值张量。
        :param pred_var: 模型预测的方差张量。方差必须为正。
        :param target: 真实的目标值张量。
        :return: 计算出的标量损失值。
        """
        # PyTorch 的 GaussianNLLLoss 要求输入(input)是预测的均值
        # 其函数签名为: GaussianNLLLoss(input, target, var)
        return self.nll_loss(pred_mean, target, pred_var)

class MSEloss(nn.Module):
    """
    均方误差损失函数。
    """
    def __init__(self, **kwargs):
        super().__init__()
        # 过滤掉 MSELoss 不支持的参数
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['reduction', 'size_average', 'reduce']}
        self.mse_loss = nn.MSELoss(**valid_kwargs)

    def forward(self, pred, target):
        return self.mse_loss(pred, target)

class MAEloss(nn.Module):
    """
    平均绝对误差损失函数。
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.mae_loss = nn.L1Loss(**kwargs)

    def forward(self, pred, target):
        return self.mae_loss(pred, target)

class InitialLoss(nn.Module):
    """
    一个用于约束波形初始阶段稳定性的损失函数。

    该损失函数惩罚预测波形在最开始一小段时间内（由 percentage 定义）
    偏离零的程度，鼓励模型生成更符合物理现实的、平稳启动的波形。
    """
    def __init__(self, percentage=0.05, weight_target=0, loss_type='mae', reduction='mean'):
        """
        :param percentage: 一个 0 到 1 之间的小数，定义了需要约束的波形初始部分的比例。
                           默认为 0.05，即前 5% 的时间点。
        :param weight_target: 目标损失的权重。如果为 0，则不计算目标损失。
        :param loss_type: 使用的损失类型，'mae' (L1) 或 'mse' (L2)。
                          MAE 对异常值更鲁棒，通常是更好的选择。
        :param reduction: 指定如何聚合损失，'mean' 或 'sum'。
        """
        super().__init__()
        if not 0 < percentage <= 1:
            raise ValueError("`percentage` 必须在 (0, 1] 范围内。")
        if loss_type not in ['mae', 'mse']:
            raise ValueError("`loss_type` 必须是 'mae' 或 'mse'。")
            
        self.percentage = percentage
        self.weight_target = weight_target
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(self, pred, target=None):
        """
        计算初始段损失。

        :param pred: 模型的预测输出张量, 形状 (B, C, L)。
        :param target: 目标张量 (在此损失中未使用，但保留以保持接口一致性)。
        :return: 计算出的标量损失值。
        """
        # --- 1. 计算需要约束的时间点数量 ---
        seq_len = pred.shape[-1]
        num_points_to_penalize = int(seq_len * self.percentage)

        if num_points_to_penalize == 0:
            # 如果信号太短或百分比太小，不计算损失
            return torch.tensor(0.0, device=pred.device)

        # --- 2. 提取波形的初始部分 ---
        initial_segment = pred[:, :, :num_points_to_penalize]

        # --- 3. 计算该部分与零的差异 ---
        if self.loss_type == 'mae':
            loss = torch.abs(initial_segment)
        else: # mse
            loss = torch.pow(initial_segment, 2)

        # --- 3.5 计算该部分与目标的差异 (可选) , 因为有时候真值波形初始部分不一定完全在0附近 ---
        if target is not None:
            initial_target_segment = target[:, :, :num_points_to_penalize]
            if self.loss_type == 'mae':
                target_loss = torch.abs(initial_target_segment)
            else:
                target_loss = torch.pow(initial_target_segment, 2)
            loss = loss + self.weight_target * target_loss

        # --- 4. 聚合损失 ---
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"不支持的 reduction 类型: {self.reduction}")

class TerminalLoss(nn.Module):
    """
    一个用于约束波形终端阶段稳定性的损失函数。

    该损失函数惩罚预测波形在最后一段时间内（由 percentage 定义）
    偏离零的程度，用于抑制模型在序列末端产生不切实际的跳变。
    """
    def __init__(self, percentage=0.05, weight_target=1.0, loss_type='mae', reduction='mean'):
        """
        :param percentage: 一个 0 到 1 之间的小数，定义了需要约束的波形末尾部分的比例。
                           默认为 0.05，即最后 5% 的时间点。
        :param weight_target: 目标损失的权重。如果为 0，则不计算目标损失。
        :param loss_type: 使用的损失类型，'mae' (L1) 或 'mse' (L2)。
        :param reduction: 指定如何聚合损失，'mean' 或 'sum'。
        """
        super().__init__()
        if not 0 < percentage <= 1:
            raise ValueError("`percentage` 必须在 (0, 1] 范围内。")
        if loss_type not in ['mae', 'mse']:
            raise ValueError("`loss_type` 必须是 'mae' 或 'mse'。")
            
        self.percentage = percentage
        self.weight_target = weight_target
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(self, pred, target=None):
        """
        计算终端段损失。

        :param pred: 模型的预测输出张量, 形状 (B, C, L)。
        :param target: 目标张量 (在此损失中未使用)。
        :return: 计算出的标量损失值。
        """
        # --- 1. 计算需要约束的时间点数量 ---
        seq_len = pred.shape[-1]
        num_points_to_penalize = int(seq_len * self.percentage)

        if num_points_to_penalize == 0:
            return torch.tensor(0.0, device=pred.device)

        # --- 2. 提取波形的末尾部分 ---
        # 使用负索引来从后往前切片
        terminal_segment = pred[:, :, -num_points_to_penalize:]

        # --- 3. 计算该部分与零的差异 ---
        if self.loss_type == 'mae':
            loss = torch.abs(terminal_segment)
        else: # mse
            loss = torch.pow(terminal_segment, 2)

        # --- 3.5 计算该部分与目标的差异 (可选)，因为有时候真值波形末尾不一定完全在0附近 ---
        if target is not None:
            terminal_target_segment = target[:, :, -num_points_to_penalize:]
            if self.loss_type == 'mae':
                target_loss = torch.abs(terminal_target_segment)
            else:
                target_loss = torch.pow(terminal_target_segment, 2)
            loss = loss + self.weight_target * target_loss

        # --- 4. 聚合损失 ---
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"不支持的 reduction 类型: {self.reduction}")

class VelocityLoss(nn.Module):
    """
    速度一致性损失函数（约束一重积分）。
    通过计算预测加速度与真实加速度的累积和（速度变化量）之间的差异，
    强制模型遵循动量守恒等物理规律，减少低频漂移。
    """
    def __init__(self, dt=0.001, loss_type='mse', reduction='mean'):
        """
        :param dt: 采样时间间隔，默认为 0.001s (1ms)。
        :param loss_type: 'l1' (MAE) 或 'mse' (L2)
        :param reduction: 'mean' 或 'sum'。
        """
        super().__init__()
        self.dt = dt
        self.reduction = reduction
        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"不支持的 loss_type: {loss_type}")

    def forward(self, pred, target):
        """
        :param pred: 预测加速度 (B, C, L)
        :param target: 真实加速度 (B, C, L)
        注意：即便输入是归一化后的加速度，对其积分的一致性约束依然有效。
        """
        # 计算速度变化量 (Delta V)
        # dim=-1 表示沿时间轴积分
        pred_vel = torch.cumsum(pred, dim=-1) * self.dt
        target_vel = torch.cumsum(target, dim=-1) * self.dt
        
        return self.criterion(pred_vel, target_vel)

# 基于ISO-Rating的Loss设计
class CorridorLoss(nn.Module):
    """
    一个可微分的代理损失函数，用于模拟 ISO 标准中的廊道评分
    该损失函数惩罚那些超出以内廊道为边界的预测值
    损失的大小与超出部分的量和指数因子相关
    """
    def __init__(self, inner_corridor_width=0.05, exponent=2.0, reduction='mean'):
        """
        :param inner_corridor_width: 对应 ISO 标准中的 a0，定义了零惩罚区的相对宽度。
                                     默认值为 0.05，与 ISO 标准一致。
        :param exponent: 对应 ISO 标准中的 kz，定义了惩罚的指数。
                         默认值为 2.0，与 ISO 标准的二次方衰减一致。
        :param reduction: 指定如何聚合损失，'mean' 或 'sum'。
        """
        super().__init__()
        self.inner_corridor_width = inner_corridor_width
        self.exponent = exponent
        self.reduction = reduction

    def forward(self, pred, target):
        """
        计算廊道损失。

        :param pred: 模型的预测输出张量, 形状 (B, C, L)。
        :param target: 真实的目标张量, 形状 (B, C, L)。
        :return: 计算出的标量损失值。
        """
        # --- 1. 计算内廊道边界 (delta_i)，与 ISO 公式 (6.3) 和 (6.4) 对应 ---
        # 为了稳定性，在 batch 维度上独立计算每个样本的 t_norm
        # keepdim=True 确保 t_norm 的形状为 (B, C, 1)，以便进行广播
        t_norm = torch.max(torch.abs(target), dim=-1, keepdim=True)[0] # 当 torch.max 带有 dim 参数时，它返回一个包含两个张量的元组：(values, indices); Shape (B, C, 1)
        
        # 增加一个小的 epsilon 防止 t_norm 为零导致 delta_i 为零
        delta_i = self.inner_corridor_width * (t_norm + 1e-9)

        # --- 2. 计算超出内廊道的误差 ---
        # 逐点绝对误差
        abs_diff = torch.abs(pred - target)
        
        # 使用 ReLU 函数，仅保留超出内廊道边界 (delta_i) 的误差部分
        # 这模拟了 ISO 标准中 "在内廊道内得分为1（即损失为0）" 的逻辑
        exceeded_error = F.relu(abs_diff - delta_i)

        # --- 3. 计算最终损失，与 ISO 公式 (6.8) 的惩罚思想对应 ---
        # 对超出的误差部分进行指数惩罚
        loss = torch.pow(exceeded_error, self.exponent)

        # --- 4. 聚合损失 ---
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"不支持的 reduction 类型: {self.reduction}")
        
class SlopeLoss(nn.Module):
    """
    一个可微分的代理损失函数，用于模拟 ISO 标准中的斜率评分。

    该损失函数通过比较预测信号和目标信号的一阶离散导数，
    来惩罚两者在局部变化趋势（即波形拓扑结构）上的差异。
    此版本增加了可配置的移动平均平滑功能，以模拟ISO标准处理流程。
    """
    def __init__(self, reduction='mean', apply_smoothing=True, smoothing_window_size=9):
        """
        :param reduction: 指定如何聚合损失，'mean' 或 'sum'。
        :param apply_smoothing: 是否对导数应用平滑处理。
        :param smoothing_window_size: 移动平均的窗口大小，应为奇数。默认为9，与ISO标准一致。
        """
        super().__init__()
        self.reduction = reduction
        self.apply_smoothing = apply_smoothing
        if self.apply_smoothing:
            if smoothing_window_size % 2 == 0:
                raise ValueError("`smoothing_window_size` 必须是奇数。")
            self.smoothing_window_size = smoothing_window_size
            
            # 创建一个不可训练的卷积核用于移动平均
            kernel = torch.ones(1, 1, self.smoothing_window_size) / self.smoothing_window_size
            self.register_buffer('smoothing_kernel', kernel)

    def forward(self, pred, target):
        """
        计算斜率损失。

        :param pred: 模型的预测输出张量, 形状 (B, C, L)。
        :param target: 真实的目标张量, 形状 (B, C, L)。
        :return: 计算出的标量损失值。
        """
        # --- 1. 计算一阶离散导数 ---
        pred_slope = pred[:, :, 1:] - pred[:, :, :-1]
        target_slope = target[:, :, 1:] - target[:, :, :-1]

        # --- 2. (可选) 对导数曲线进行平滑处理 ---
        if self.apply_smoothing:
            batch_size, num_channels, seq_len = pred_slope.shape
            
            pred_slope_reshaped = pred_slope.view(batch_size * num_channels, 1, seq_len)
            target_slope_reshaped = target_slope.view(batch_size * num_channels, 1, seq_len)

            # +++ 将卷积核动态移动到与输入张量相同的设备上 +++
            kernel = self.smoothing_kernel.to(pred.device)

            smoothed_pred_slope = F.conv1d(pred_slope_reshaped, kernel, padding='same')
            smoothed_target_slope = F.conv1d(target_slope_reshaped, kernel, padding='same')
            
            pred_slope = smoothed_pred_slope.view(batch_size, num_channels, seq_len)
            target_slope = smoothed_target_slope.view(batch_size, num_channels, seq_len)

        # --- 3. 计算导数之间的误差 ---
        loss = F.mse_loss(pred_slope, target_slope, reduction='none')

        # --- 4. 聚合损失 ---
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"不支持的 reduction 类型: {self.reduction}")

class PhaseLoss(nn.Module):
    """
    一个可微分的代理损失函数，用于模拟 ISO 标准中的相位评分。
    """
    def __init__(self, n_fft=64, hop_length=16, win_length=64, reduction='mean'):
        """
        :param n_fft: FFT 的点数，控制频率分辨率。
        :param hop_length: 帧移长度，控制时间分辨率。
        :param win_length: 窗函数长度。
        :param reduction: 指定如何聚合损失，'mean' 或 'sum'。
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.reduction = reduction
        self.register_buffer('window', torch.hann_window(self.win_length))

    def forward(self, pred, target):
        """
        计算相位损失。
        """
        window = self.window.to(pred.device)

        batch_size, num_channels, seq_len = pred.shape
        
        pred_reshaped = pred.view(batch_size * num_channels, seq_len)
        target_reshaped = target.view(batch_size * num_channels, seq_len)

        pred_stft = torch.stft(pred_reshaped, n_fft=self.n_fft, hop_length=self.hop_length,
                               win_length=self.win_length, window=window,
                               return_complex=True, center=True)
        
        target_stft = torch.stft(target_reshaped, n_fft=self.n_fft, hop_length=self.hop_length,
                                 win_length=self.win_length, window=window,
                                 return_complex=True, center=True)

        # --- 2. 计算复数谱图之间的误差 ---
        # F.mse_loss 不支持复数张量，我们手动计算差值幅度的平方
        diff = pred_stft - target_stft
        # .abs() 计算复数的模 (magnitude)
        # .pow(2) 计算模的平方，这在数学上等价于复数的 MSE
        loss = diff.abs().pow(2)

        # --- 3. 聚合损失 ---
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"不支持的 reduction 类型: {self.reduction}")

