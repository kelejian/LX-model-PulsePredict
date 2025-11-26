import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import model.loss as module_loss
from torch_geometric.nn import MLP as PygMLP # 直接用PyG的MLP模块

# ==========================================================================================
# 基础组件定义 (Basic Components)
# ==========================================================================================

class ResMLPBlock(nn.Module):
    """
    残差 MLP 块
    
    用途: 用于深度编码器，增加网络深度的同时防止梯度消失和模型退化
    结构: Linear -> BN -> SiLU -> Dropout -> Linear -> BN -> SiLU -> Dropout + 残差连接
    
    参数:
        hidden_dim: 隐藏层维度
        dropout: Dropout 概率
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        输入: (B, hidden_dim)
        输出: (B, hidden_dim)
        """
        return x + self.block(x)

class SeedFeatureProjector(nn.Module):
    """
    高效种子特征投影器
    
    用途: 替代巨大的全连接层，先将全局特征映射到低维流形，再通过卷积混合到目标通道
    策略: 低秩分解 - 降低参数量的同时保持表达能力
    结构: Linear -> Reshape -> BN -> SiLU -> Dropout -> Conv1x1
    
    参数:
        z_dim: 输入全局特征维度
        output_len: 输出序列长度
        output_channels: 输出通道数
        proj_channels: 中间投影通道数（低秩维度）
        dropout: Dropout 概率
    """
    def __init__(self, z_dim, output_len, output_channels, proj_channels=32, dropout=0.1):
        super().__init__()
        self.output_len = output_len
        self.proj_channels = proj_channels
        
        # 低秩线性映射
        self.linear = nn.Linear(z_dim, output_len * proj_channels, bias=False)
        self.bn = nn.BatchNorm1d(proj_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # 通道混合层
        self.mixer = nn.Conv1d(proj_channels, output_channels, kernel_size=1, bias=True)

    def forward(self, z):
        """
        输入: (B, z_dim)
        输出: (B, output_channels, output_len)
        """
        B = z.shape[0]
        # 低秩线性映射: (B, z_dim) -> (B, output_len * proj_channels)
        x = self.linear(z)
        # 重塑为序列形式: (B, proj_channels, output_len)
        x = x.view(B, self.proj_channels, self.output_len)
        
        # 归一化与激活
        x = self.act(self.bn(x))
        x = self.dropout(x)
        
        # 通道混合: (B, proj_channels, output_len) -> (B, output_channels, output_len)
        x = self.mixer(x)
        return x

class BiGRUBottleneck(nn.Module):
    """
    双向 GRU 时序瓶颈
    
    用途: 在低分辨率下注入时序演化逻辑，建立特征间的因果依赖关系
    结构: Bi-GRU -> Linear -> LayerNorm -> SiLU
    
    参数:
        input_dim: 输入特征维度
        hidden_dim: GRU 隐藏层维度
        output_dim: 输出特征维度
        gru_layers: GRU 层数
    """
    def __init__(self, input_dim, hidden_dim, output_dim, gru_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            bias=True  # GRU 门控机制需要 bias
        )
        # 双向 GRU 输出维度为 hidden_dim * 2
        self.proj = nn.Linear(hidden_dim * 2, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """
        输入: (B, input_dim, L)
        输出: (B, output_dim, L)
        """
        # 转换为 GRU 输入格式: (B, input_dim, L) -> (B, L, input_dim)
        x = x.permute(0, 2, 1)
        
        # GRU 处理: (B, L, input_dim) -> (B, L, hidden_dim*2)
        gru_out, _ = self.gru(x)
        
        # 投影: (B, L, hidden_dim*2) -> (B, L, output_dim)
        out = self.proj(gru_out)
        
        # Post-Norm 与激活
        out = self.norm(out)
        out = self.act(out)
        
        # 转换回通道优先格式: (B, L, output_dim) -> (B, output_dim, L)
        return out.permute(0, 2, 1)

class PixelShuffle1D(nn.Module):
    """
    一维亚像素卷积上采样（PixelShuffle）
    
    用途: 将通道维度重排为时间维度，实现高效上采样
    原理: (B, C*r, L) -> (B, C, L*r)，其中 r 为上采样倍率
    
    参数:
        upscale_factor: 上采样倍率
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        """
        输入: (B, C*upscale_factor, L)
        输出: (B, C, L*upscale_factor)
        """
        batch_size, channels, steps = x.size()
        r = self.upscale_factor
        
        if channels % r != 0:
            raise ValueError(f"输入通道数 {channels} 必须能被上采样倍率 {r} 整除")
        
        new_channels = channels // r
        
        # 重排: (B, C_out*r, L) -> (B, C_out, r, L) -> (B, C_out, L, r) -> (B, C_out, L*r)
        x = x.view(batch_size, new_channels, r, steps)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(batch_size, new_channels, steps * r)
        return x

def icnr_init(conv_layer, upscale_factor, initializer=nn.init.kaiming_normal_):
    """
    ICNR 初始化 (针对 1D 卷积)
    
    原理: 将权重初始化为一种特殊形态，使得 Conv1d + PixelShuffle 最初的行为等价于 Nearest Neighbor Upsampling。
    
    参数:
        conv_layer: 需要初始化的 nn.Conv1d 层
        upscale_factor: 上采样倍率
        initializer: 基础初始化函数 (默认 Kaiming Normal)
    """
    w = conv_layer.weight.data
    out_channels, in_channels, kernel_size = w.shape
    
    # 1. 计算“种子”权重的输出通道数 (C_out / r)
    if out_channels % upscale_factor != 0:
        raise ValueError("输出通道数必须能被上采样倍率整除")
        
    sub_kernel_out = out_channels // upscale_factor
    
    # 2. 生成低维种子权重 (shape: [C_out/r, C_in, K])
    kernel_shape = (sub_kernel_out, in_channels, kernel_size)
    w_seed = torch.zeros(kernel_shape)
    initializer(w_seed) # 使用标准方法初始化种子
    
    # 3. 沿输出通道维度进行复制扩展 (Repeat Interleave)
    # 效果: [w1, w2] -> [w1, w1, ..., w2, w2, ...] (每个重复 r 次)
    # 结合 PixelShuffle 的重排逻辑 (view -> permute -> flatten)，这确保了
    # 相邻的 r 个子像素拥有相同的权重，从而输出相同的值 (即最近邻插值)。
    w_new = w_seed.repeat_interleave(upscale_factor, dim=0)
    
    # 4. 赋值回权重
    conv_layer.weight.data.copy_(w_new)
    
    # 5. 处理 Bias: 建议初始化为 0，避免初始阶段引入通道间的直流偏差
    if conv_layer.bias is not None:
        nn.init.zeros_(conv_layer.bias)

class AdapLengthAlign1D(nn.Module):
    """
    自适应长度对齐层
    
    用途: 确保序列长度严格对齐到目标长度，处理上采样后的长度偏差
    策略:
        - 头部锚定: 保护 t=0 物理起点，不进行左侧裁剪/填充
        - 尾部填充: 使用 'replicate' 模式复制边界值，保持数值连续性
        - 尾部裁剪: 直接截断多余部分
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, target_length):
        """
        输入: (B, C, curr_length)
        输出: (B, C, target_length)
        """
        curr_length = x.shape[-1]
        diff = target_length - curr_length

        if diff == 0:
            return x
        
        if diff > 0:
            # 尾部填充: 右侧复制填充 diff 个时间步
            return F.pad(x, (0, diff), mode='replicate')
        else:
            # 尾部裁剪: 截断至目标长度
            return x[..., :target_length]

class ContextInjection(nn.Module):
    """
    上下文注入模块
    
    用途: 将全局工况特征向量注入到多尺度特征图中
    方法: 通道拼接 + 1x1 卷积融合
    结构: Concat -> Conv1x1 -> BN -> SiLU
    
    参数:
        feature_channels: 特征图通道数
        z_dim: 全局特征维度
    """
    def __init__(self, feature_channels, z_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv1d(feature_channels + z_dim, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(feature_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x, z_prime):
        """
        输入:
            x: (B, feature_channels, L) - 特征图
            z_prime: (B, z_dim) - 全局特征
        输出: (B, feature_channels, L)
        """
        B, C, L = x.shape
        # 广播全局特征: (B, z_dim) -> (B, z_dim, L)
        z_expanded = z_prime.unsqueeze(2).expand(-1, -1, L)
        # 拼接并融合: (B, feature_channels+z_dim, L) -> (B, feature_channels, L)
        combined = torch.cat([x, z_expanded], dim=1)    
        return self.fusion(combined)

class ResBlock1D(nn.Module):
    """
    一维残差块
    
    用途: 提取局部时序特征，通过残差连接增强梯度流动
    结构: Conv -> BN -> SiLU -> Conv -> BN + 残差连接 -> SiLU
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 通道数不匹配时使用 1x1 卷积调整
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        """
        输入: (B, in_channels, L)
        输出: (B, out_channels, L)
        """
        residual = self.shortcut(x)
        out = F.silu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.silu(out, inplace=True)

class DeepRegressionHead(nn.Module):
    """
    深层回归头
    
    用途: 解码最终波形输出
    设计原则: 移除 BN 和 Dropout，保证回归数值的绝对尺度和连续性
    结构: Conv3x3 -> SiLU -> Conv3x3 -> SiLU -> Conv1x1
    
    参数:
        in_channels: 输入特征通道数
        out_channels: 输出通道数（1 或 2，取决于是否使用高斯 NLL）
        hidden_dim: 隐藏层通道数
    """
    def __init__(self, in_channels, out_channels=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=True) 
        )

    def forward(self, x):
        """
        输入: (B, in_channels, L)
        输出: (B, out_channels, L)
        """
        return self.net(x)

# ==========================================================================================
# 主模型定义 (Hybrid PulseCNN)
# ==========================================================================================

class HybridPulseCNN(BaseModel):
    """
    主模型定义
    架构: Deep Residual MLP -> Bi-GRU Bottleneck -> Trident Progressive Decoder

    参数:
        input_dim: 输入工况特征维度
        output_channels: 输出通道数（xyz 三轴）
        mlp_hidden_dim: MLP 编码器隐藏层维度
        seed_proj_channels: 种子投影器中间通道数
        gru_hidden_dim: GRU 隐藏层维度
        gru_layers: GRU 层数
        channel_configs: 各解码阶段的通道配置列表 [Stage1, Stage2, Stage3]
        output_lengths: 各解码阶段的输出长度列表 [L1, L2, L3]
        GauNll_use: 是否使用高斯负对数似然损失（输出均值和方差）
    """
    def __init__(self, input_dim=3, output_channels=3, 
                 mlp_hidden_dim=384, 
                 seed_proj_channels=32,
                 gru_hidden_dim=128,
                 gru_layers=1,
                 channel_configs=[128, 64, 32],
                 output_lengths=[37, 75, 150],
                 GauNll_use=True):
        super().__init__()
        
        self.GauNll_use = GauNll_use
        self.output_lengths = output_lengths
        self.channel_configs = channel_configs

        # 自动计算各阶段上采样倍率
        self.upscale_factors = []
        for i in range(len(output_lengths) - 1):
            factor = output_lengths[i+1] / output_lengths[i]
            r = max(1, int(round(factor))) 
            self.upscale_factors.append(r)

        # ========================================================================
        # 1. 深度流形编码器
        # ========================================================================
        self.encoder_input = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim, bias=False),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.encoder_body = nn.Sequential(
            ResMLPBlock(mlp_hidden_dim, dropout=0.1),
            ResMLPBlock(mlp_hidden_dim, dropout=0.1)
        )
        self.z_dim = mlp_hidden_dim

        # ========================================================================
        # 2. 序列初始化与瓶颈
        # ========================================================================
        self.init_len = output_lengths[0]
        self.gru_input_dim = channel_configs[0]
        
        # 高效种子投影器
        self.seed_projector = SeedFeatureProjector(
            z_dim=mlp_hidden_dim,
            output_len=self.init_len,
            output_channels=self.gru_input_dim,
            proj_channels=seed_proj_channels, 
            dropout=0.1
        )

        # Bi-GRU 时序瓶颈
        self.bottleneck = BiGRUBottleneck(
            input_dim=self.gru_input_dim,
            hidden_dim=gru_hidden_dim,
            output_dim=self.gru_input_dim,
            gru_layers=gru_layers
        )
        
        # 长度对齐模块
        self.length_align = AdapLengthAlign1D()

        # ========================================================================
        # 3. Stage 1: 共享解码器（低分辨率）
        # ========================================================================
        self.s1_context = ContextInjection(channel_configs[0], self.z_dim)
        self.s1_resblock = ResBlock1D(channel_configs[0], channel_configs[0])
        
        # 辅助输出头（用于多尺度监督）
        head_out_dim = 2 if GauNll_use else 1
        self.s1_head = DeepRegressionHead(channel_configs[0], out_channels=output_channels * head_out_dim)

        # ========================================================================
        # 4. Stage 2: 三叉分支（中等分辨率）
        # ========================================================================
        self.s2_branches = nn.ModuleDict()
        r1 = self.upscale_factors[0]
        
        for axis in ['x', 'y', 'z']:
            layers = nn.ModuleDict()
            
            # 上采样层
            layers['up_conv'] = nn.Conv1d(channel_configs[0], channel_configs[1] * r1, kernel_size=1, bias=True)
            icnr_init(layers['up_conv'], upscale_factor=r1) # ICNR 初始化
            layers['pixel_shuffle'] = PixelShuffle1D(upscale_factor=r1)
            
            # 平滑卷积层
            layers['smooth_conv'] = nn.Conv1d(channel_configs[1], channel_configs[1], kernel_size=3, padding='same', bias=False)
            layers['smooth_bn'] = nn.BatchNorm1d(channel_configs[1])
            layers['smooth_act'] = nn.SiLU(inplace=True)

            # 精炼模块
            layers['context'] = ContextInjection(channel_configs[1], self.z_dim)
            layers['resblock'] = ResBlock1D(channel_configs[1], channel_configs[1])
            layers['head'] = DeepRegressionHead(channel_configs[1], out_channels=head_out_dim)
            self.s2_branches[axis] = layers

        # ========================================================================
        # 5. Stage 3: 独立精炼（高分辨率）
        # ========================================================================
        self.s3_branches = nn.ModuleDict()
        r2 = self.upscale_factors[1]

        for axis in ['x', 'y', 'z']:
            layers = nn.ModuleDict()

            # 上采样层
            layers['up_conv'] = nn.Conv1d(channel_configs[1], channel_configs[2] * r2, kernel_size=1, bias=True)
            icnr_init(layers['up_conv'], upscale_factor=r1) # ICNR 初始化
            layers['pixel_shuffle'] = PixelShuffle1D(upscale_factor=r2)
            
            # 平滑卷积层
            layers['smooth_conv'] = nn.Conv1d(channel_configs[2], channel_configs[2], kernel_size=3, padding='same', bias=False)
            layers['smooth_bn'] = nn.BatchNorm1d(channel_configs[2])
            layers['smooth_act'] = nn.SiLU(inplace=True)

            # 精炼模块
            layers['context'] = ContextInjection(channel_configs[2], self.z_dim)
            layers['resblock'] = ResBlock1D(channel_configs[2], channel_configs[2])
            layers['head'] = DeepRegressionHead(channel_configs[2], out_channels=head_out_dim)
            self.s3_branches[axis] = layers

    def forward(self, x):
        """
        前向传播
        
        输入: (B, input_dim)
        输出: [(s1_mean, s1_var), (s2_mean, s2_var), (s3_mean, s3_var)]
              或 [s1_mean, s2_mean, s3_mean] (若 GauNll_use=False)
        
        其中:
            s1: (B, output_channels, output_lengths[0])
            s2: (B, output_channels, output_lengths[1])
            s3: (B, output_channels, output_lengths[2])
        """
        B = x.size(0)
        
        # ====================================================================
        # 1. 编码阶段: 提取全局特征
        # ====================================================================
        # (B, input_dim) -> (B, mlp_hidden_dim)
        z_prime = self.encoder_input(x)
        z_prime = self.encoder_body(z_prime)
        
        # ====================================================================
        # 2. 瓶颈阶段: 生成初始序列
        # ====================================================================
        # (B, mlp_hidden_dim) -> (B, channel_configs[0], output_lengths[0])
        seed_seq = self.seed_projector(z_prime) 
        # Bi-GRU 精炼: (B, channel_configs[0], output_lengths[0]) -> (B, channel_configs[0], output_lengths[0])
        f_s1_in = self.bottleneck(seed_seq)

        # ====================================================================
        # 3. Stage 1 解码: 共享低分辨率特征
        # ====================================================================
        # 上下文注入与残差精炼
        f_s1 = self.s1_context(f_s1_in, z_prime)
        f_s1 = self.s1_resblock(f_s1)
        
        # 输出预测: (B, channel_configs[0], output_lengths[0]) -> (B, output_channels*(1or2), output_lengths[0])
        s1_out = self.s1_head(f_s1)
        s1_tuple = self._process_head_output(s1_out)

        # ====================================================================
        # 4. Stage 2 解码: 三叉戟中分辨率分支
        # ====================================================================
        s2_preds_list = []
        f_s2_feats = {}
        
        for axis in ['x', 'y', 'z']:
            layers = self.s2_branches[axis]
            
            # 上采样: (B, channel_configs[0], output_lengths[0]) -> 
            #         (B, channel_configs[1]*r1, output_lengths[0]) -> 
            #         (B, channel_configs[1], output_lengths[0]*r1)
            feat = layers['up_conv'](f_s1)      
            feat = layers['pixel_shuffle'](feat) 
            
            # 长度对齐: (B, channel_configs[1], ~output_lengths[1]) -> (B, channel_configs[1], output_lengths[1])
            feat = self.length_align(feat, self.output_lengths[1])
            
            # 平滑融合: 卷积扫过对齐边界，消除填充/裁剪伪影
            feat = layers['smooth_act'](layers['smooth_bn'](layers['smooth_conv'](feat)))
            
            # 上下文注入与残差精炼
            feat = layers['context'](feat, z_prime)
            feat = layers['resblock'](feat)
            
            # 缓存特征用于下一阶段
            f_s2_feats[axis] = feat
            # 输出预测: (B, channel_configs[1], output_lengths[1]) -> (B, (1or2), output_lengths[1])
            s2_preds_list.append(layers['head'](feat))

        # 拼接三轴输出: (B, output_channels*(1or2), output_lengths[1])
        s2_out = torch.cat(s2_preds_list, dim=1)
        s2_tuple = self._process_head_output(s2_out)

        # ====================================================================
        # 5. Stage 3 解码: 独立高分辨率精炼
        # ====================================================================
        s3_preds_list = []
        
        for axis in ['x', 'y', 'z']:
            layers = self.s3_branches[axis]
            prev_feat = f_s2_feats[axis]
            
            # 上采样: (B, channel_configs[1], output_lengths[1]) -> 
            #         (B, channel_configs[2]*r2, output_lengths[1]) -> 
            #         (B, channel_configs[2], output_lengths[1]*r2)
            feat = layers['up_conv'](prev_feat)
            feat = layers['pixel_shuffle'](feat)
            
            # 长度对齐: (B, channel_configs[2], ~output_lengths[2]) -> (B, channel_configs[2], output_lengths[2])
            feat = self.length_align(feat, self.output_lengths[2])
            
            # 平滑融合
            feat = layers['smooth_act'](layers['smooth_bn'](layers['smooth_conv'](feat)))
            
            # 上下文注入与残差精炼
            feat = layers['context'](feat, z_prime)
            feat = layers['resblock'](feat)
            
            # 输出预测: (B, channel_configs[2], output_lengths[2]) -> (B, (1or2), output_lengths[2])
            s3_preds_list.append(layers['head'](feat))

        # 拼接三轴输出: (B, output_channels*(1or2), output_lengths[2])
        s3_out = torch.cat(s3_preds_list, dim=1)
        s3_tuple = self._process_head_output(s3_out)

        return [s1_tuple, s2_tuple, s3_tuple]

    def _process_head_output(self, raw_out):
        """
        处理回归头输出，分离均值和方差
        
        输入: (B, output_channels*k, L)，k=1（仅均值）或 k=2（均值+方差）
        输出: 
            - GauNll_use=True: (mean, var)，每个形状为 (B, output_channels, L)
            - GauNll_use=False: raw_out，形状为 (B, output_channels, L)
        """
        if self.GauNll_use:
            # 通道布局: [Mx, Vx, My, Vy, Mz, Vz] -> 重塑为 [x:(M,V), y:(M,V), z:(M,V)]
            B, C, L = raw_out.shape
            reshaped = raw_out.view(B, 3, 2, L)
            mean = reshaped[:, :, 0, :]      # (B, output_channels, L)
            log_var = reshaped[:, :, 1, :]   # (B, output_channels, L)
            var = torch.exp(log_var)         # 转换为方差
            return (mean, var)
        else:
            return raw_out

    # =========================================================================
    # 损失与评估接口
    # =========================================================================
    def compute_loss(self, model_output, target, criterions):
        """
        计算模型损失
        
        参数:
            model_output: 模型输出列表 [(s1_mean, s1_var), (s2_mean, s2_var), (s3_mean, s3_var)]
            target: 目标波形 (B, output_channels, L_target)
            criterions: 损失函数配置列表
        
        返回:
            total_loss: 加权总损失
            loss_components: 各损失分量字典
        """
        total_loss = torch.tensor(0.0, device=target.device)
        loss_components = {}

        for criterion_item in criterions:
            loss_instance = criterion_item['instance']
            weight = criterion_item['weight']
            channel_weights = criterion_item['channel_weights']
            loss_type_name = type(loss_instance).__name__

            current_loss_item = torch.tensor(0.0, device=target.device)
            for i in range(len(channel_weights)):
                if channel_weights[i] == 0: continue
                target_channel = target[:, i:i+1, :]

                if hasattr(loss_instance, 'loss_weights'): # 多尺度损失
                    if self.GauNll_use:
                        pred_channel = [(stage_out[0][:, i:i+1, :], stage_out[1][:, i:i+1, :]) for stage_out in model_output]
                    else:
                        pred_channel = [stage_out[:, i:i+1, :] for stage_out in model_output]

                    target_list_channel = []
                    for stage_pred in pred_channel:
                        curr_len = stage_pred[0].shape[-1] if isinstance(stage_pred, tuple) else stage_pred.shape[-1]
                        if curr_len == target_channel.shape[-1]:
                            t_resized = target_channel
                        else:
                            t_resized = F.interpolate(target_channel, size=curr_len, mode='linear', align_corners=False)
                        target_list_channel.append(t_resized)
                    
                    channel_loss = loss_instance(pred_channel, target_list_channel)
                else: # 单尺度损失
                    final_stage = model_output[-1]
                    pred_mean = final_stage[0][:, i:i+1, :] if self.GauNll_use else final_stage[:, i:i+1, :]
                    channel_loss = loss_instance(pred_mean, target_channel)

                current_loss_item += channel_weights[i] * channel_loss

            loss = current_loss_item / sum(channel_weights) if sum(channel_weights) > 0 else torch.tensor(0.0, device=target.device)
            total_loss += weight * loss
            loss_components[loss_type_name] = loss.item()

        return total_loss, loss_components

    def get_metrics_output(self, model_output):
        """
        提取用于评估指标的模型输出
        
        返回: 最终阶段的均值预测 (B, output_channels, output_lengths[-1])
        """
        return model_output[-1][0] if self.GauNll_use else model_output[-1]

# ==========================================================================================
# 之前的 PulseCNN 模型定义
# ==========================================================================================
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
class PulseCNN(BaseModel):
    """
    基于1D-CNN的碰撞波形预测模型。
    该模型采用层级式多尺度生成架构，其上采样单元为残差精调模块，
    旨在实现从粗到精的“逐步细化预测”。
    """
    def __init__(self, input_dim=3, output_channels=3,
                 mlp_latent_dim=256, mlp_num_layers=3, mlp_dropout=0.1, 
                 projection_init_channels=16, projection_dropout=0.2,
                 UpLayer_ks=3, ResLayer_ks=3,
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
            num_layers=mlp_num_layers, norm="batch_norm", act="leaky_relu", plain_last=False, 
            dropout=mlp_dropout
        )

        # 2. 投影块
        self.projection_block = nn.Sequential(
            nn.Linear(mlp_latent_dim + input_dim, self.initial_length * projection_init_channels),
            nn.Dropout(projection_dropout),
            nn.BatchNorm1d(self.initial_length * projection_init_channels), 
            nn.LeakyReLU(),
            # 将线性层的输出重塑为 (B, C, L)
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
        proj_out = self.projection_block[0](z_prime) # linear
        proj_out = self.projection_block[1](proj_out) # dropout
        proj_out = self.projection_block[2](proj_out) # batchnorm
        proj_out = self.projection_block[3](proj_out) # relu
        proj_out_reshaped = proj_out.view(-1, self.projection_init_channels, self.initial_length) # reshape
        initial_sequence = self.projection_block[4](proj_out_reshaped) # conv1d

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
            此版本修复了非多尺度损失（如CorridorLoss）的输入张量尺寸问题。
            返回总损失和一个包含各分量损失的字典。
            """
            total_loss = torch.tensor(0.0, device=target.device)
            loss_components = {}

            # --- 遍历所有损失函数并计算加权和 ---
            for criterion_item in criterions:
                loss_instance = criterion_item['instance']
                weight = criterion_item['weight']
                channel_weights = criterion_item['channel_weights']
                loss_type_name = type(loss_instance).__name__
                
                if not channel_weights or len(channel_weights) != target.shape[1]:
                    raise ValueError(f"'{loss_type_name}' 的 channel_weights ({channel_weights}) 必须是一个包含 {target.shape[1]} 个元素的列表。")

                current_loss_item = torch.tensor(0.0, device=target.device)
                
                # --- 统一的通道加权计算逻辑 ---
                for i in range(len(channel_weights)):
                    if channel_weights[i] == 0:
                        continue

                    target_channel = target[:, i:i+1, :] # 目标通道，长度为150

                    # --- 区分不同loss类型的计算 ---
                    if isinstance(loss_instance, module_loss.MultiLoss):
                        # 为 MultiLoss 准备多尺度的单通道预测
                        if self.GauNll_use:
                            pred_mean_list, pred_var_list = model_output
                            pred_mean_channel = [p[:, i:i+1, :] for p in pred_mean_list]
                            pred_var_channel = [v[:, i:i+1, :] for v in pred_var_list]
                            pred_channel = list(zip(pred_mean_channel, pred_var_channel))
                        else:
                            pred_mean_list = model_output
                            pred_channel = [p[:, i:i+1, :] for p in pred_mean_list]
                        
                        # 为 MultiLoss 准备多尺度的单通道目标
                        target_list_channel = [F.interpolate(target_channel, size=p[0].shape[-1] if self.GauNll_use else p.shape[-1], mode='linear', align_corners=False) for p in pred_channel]
                        target_list_channel[-1] = target_channel
                        
                        channel_loss = loss_instance(pred_channel, target_list_channel)
                    
                    else:
                        # 为 CorridorLoss 等非多尺度损失，显式地提取最终预测结果的单通道
                        if self.GauNll_use:
                            # model_output[0] 是 pred_mean_list, [-1] 是最后一个尺度的预测
                            final_pred_for_channel = model_output[0][-1][:, i:i+1, :]
                        else:
                            # model_output 是 pred_mean_list, [-1] 是最后一个尺度的预测
                            final_pred_for_channel = model_output[-1][:, i:i+1, :]
                        
                        # 此处 final_pred_for_channel 的长度是150，与 target_channel 长度一致
                        channel_loss = loss_instance(final_pred_for_channel, target_channel)

                    current_loss_item += channel_weights[i] * channel_loss
                
                loss = current_loss_item / sum(channel_weights) if sum(channel_weights) > 0 else torch.tensor(0.0, device=target.device)

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
