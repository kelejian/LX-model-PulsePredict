import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import model.loss as module_loss # 导入自身模块以便动态创建base_loss

def nll_loss(output, target):
    return F.nll_loss(output, target)

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
        self.mse_loss = nn.MSELoss(**kwargs)

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

class MultiLoss(nn.Module):
    """
    一个通用的多路损失加权求和模块。
    """
    def __init__(self, loss_weights, base_loss_type, base_loss_args=None):
        """
        :param loss_weights: 一个列表，包含每一路损失的权重。
        :param base_loss_type: 基础损失函数的类型名称 (字符串)，例如 'GaussianNLLLoss'。
        :param base_loss_args: 一个字典，包含基础损失函数的初始化参数。
        """
        super().__init__()
        if base_loss_args is None:
            base_loss_args = {}
        self.loss_weights = loss_weights
        # 使用getattr动态地从本模块(module_loss)中获取损失函数类并实例化
        self.base_loss = getattr(module_loss, base_loss_type)(**base_loss_args)
        
    def forward(self, pred_list, target_list):
        """
        计算加权总损失。

        :param pred_list: 预测值列表。
                          - 对于普通loss，每个元素是 pred_tensor。
                          - 对于GauNLL, 每个元素是 (pred_mean_tensor, pred_var_tensor) 的元组。
        :param target_list: 目标值列表，应与 pred_list 中的张量形状一一对应。
        :return: 加权后的总损失标量。
        """
        if len(self.loss_weights) != len(pred_list):
            raise ValueError(f"loss_weights (len={len(self.loss_weights)}) 和 pred_list (len={len(pred_list)}) 的长度必须一致。")
        
        total_loss = 0
        for i, (pred, target) in enumerate(zip(pred_list, target_list)):
            if isinstance(pred, tuple):
                # 适用于需要多个输入的loss，如 GaussianNLLLoss(mean, target, var)
                loss = self.base_loss(*pred, target)
            else:
                # 适用于标准loss，如 MSELoss(pred, target)
                loss = self.base_loss(pred, target)
            
            total_loss += self.loss_weights[i] * loss
            
        return total_loss / sum(self.loss_weights)

class InitialLoss(nn.Module):
    """
    一个用于约束波形初始阶段稳定性的损失函数。

    该损失函数惩罚预测波形在最开始一小段时间内（由 percentage 定义）
    偏离零的程度，鼓励模型生成更符合物理现实的、平稳启动的波形。
    """
    def __init__(self, percentage=0.05, loss_type='mae', reduction='mean'):
        """
        :param percentage: 一个 0 到 1 之间的小数，定义了需要约束的波形初始部分的比例。
                           默认为 0.05，即前 5% 的时间点。
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

        # --- 4. 聚合损失 ---
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"不支持的 reduction 类型: {self.reduction}")

# 基于ISO-Rating的Loss设计
class CorridorLoss(nn.Module):
    """
    一个可微分的代理损失函数，用于模拟 ISO 标准中的廊道评分。

    该损失函数惩罚那些超出以内廊道为边界的预测值。
    损失的大小与超出部分的量和指数因子相关，这与 ISO 标准的精神一致。
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
        t_norm = torch.max(torch.abs(target), dim=-1, keepdim=True)[0]
        
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
    """
    def __init__(self, reduction='mean'):
        """
        :param reduction: 指定如何聚合损失，'mean' 或 'sum'。
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        计算斜率损失。

        :param pred: 模型的预测输出张量, 形状 (B, C, L)。
        :param target: 真实的目标张量, 形状 (B, C, L)。
        :return: 计算出的标量损失值。
        """
        # --- 1. 计算一阶离散导数 ---
        # 通过计算相邻时间点之差来近似导数
        # pred[:, :, 1:] 表示从第二个时间点开始的所有点
        # pred[:, :, :-1] 表示从第一个时间点到倒数第二个点的所有点
        pred_slope = pred[:, :, 1:] - pred[:, :, :-1]
        target_slope = target[:, :, 1:] - target[:, :, :-1]

        # --- 2. 计算导数之间的误差 ---
        # 使用均方误差 (MSE) 来衡量两条导数曲线的差异
        # 这与 ISO 标准中比较导数曲线范数差异的思想一致
        loss = F.mse_loss(pred_slope, target_slope, reduction='none')

        # --- 3. 聚合损失 ---
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

