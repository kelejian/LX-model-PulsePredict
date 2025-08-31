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
    