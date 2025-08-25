import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from base import BaseDataLoader # 继承自项目基类
from sklearn.preprocessing import MinMaxScaler
from utils import InputScaler, PulseAbsMaxScaler


#==========================================================================================
# 定制的 Dataset 类来处理碰撞数据
#==========================================================================================
class CollisionDataset(Dataset):
    """
    用于加载碰撞波形数据的自定义数据集。
    【版本说明】: 此版本直接读取由 utils.data_utils.py 预处理和打包后的 .npz 文件，并加载其中所有案例。
    """
    def __init__(self, params_npz_path, processed_pulses_path, target_scaler=None):
        """
        :param params_npz_path: 包含所有工况参数的 npz 文件路径。
        :param processed_pulses_path: 包含预处理后波形数据的 .npz 文件路径。
        :param target_scaler: 可选的目标缩放器，用于对波形进行缩放。
        """
        self.target_scaler = target_scaler

        # 加载预处理后的波形数据，并从中获取所有原始 case_id
        self.pulses_data = np.load(processed_pulses_path)
        self.case_ids = np.array(sorted([int(k) for k in self.pulses_data.keys()]))
        
        # 加载并处理与 self.case_ids 对应的工况参数
        all_params = np.load(params_npz_path)
        case_indices = self.case_ids - 1 # 将 case_ids 转换为0基索引

        input_scaler = InputScaler(v_min=23.5, v_max=65, a_abs_max=60, o_abs_max=1)

        raw_velocities = all_params['impact_velocity'][case_indices]
        raw_angles = all_params['impact_angle'][case_indices]
        raw_overlaps = all_params['overlap'][case_indices]

        norm_velocities, norm_angles, norm_overlaps = input_scaler.transform(
            raw_velocities, raw_angles, raw_overlaps
        ) # 分别归一化到[0, 1]，[-1, 1]，[-1,1]

        self.features = torch.tensor(
            np.stack([norm_velocities, norm_angles, norm_overlaps], axis=1),
            dtype=torch.float32
        ) # 形状 (N, 3)，N为样本数

    def __len__(self):
        """
        返回数据集中样本的总数
        """
        return len(self.case_ids)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个样本。
        """
        input_features = self.features[idx]
        case_id = self.case_ids[idx]
        
        # 从已加载的数据中直接获取波形
        waveforms_np = self.pulses_data[str(case_id)] # 形状 (3, 200)

        if self.target_scaler is not None: # 如果提供了目标缩放器，则对波形进行缩放
            original_shape = waveforms_np.shape
            waveforms_reshaped = waveforms_np.reshape(-1, 1)
            waveforms_scaled = self.target_scaler.transform(waveforms_reshaped)
            waveforms_np = waveforms_scaled.reshape(original_shape)
            
        target_waveforms = torch.tensor(waveforms_np, dtype=torch.float32)

        return input_features, target_waveforms, case_id # 返回特征和波形数据，及其原始case_id

#==========================================================================================
#  DataLoader 类
#==========================================================================================
import joblib # 导入 joblib 用于保存和加载 scaler
class CollisionDataLoader(BaseDataLoader):
    """
    用于加载碰撞波形数据的 DataLoader 类。
    【版本说明】: 此版本实现了在训练时拟合并保存Scaler，在测试时加载Scaler的功能。
    """
    def __init__(self, params_npz_path, processed_pulses_path, batch_size, pulse_norm_mode='none', scaler_path=None, shuffle=True, validation_split=0.1, num_workers=1, training=True):
        """
        :param params_npz_path: 包含所有工况参数的 npz 文件路径。
        :param processed_pulses_path: 要加载的预处理波形数据文件路径.
        :param batch_size: 批量大小。
        :param pulse_norm_mode: 归一化模式，'none', 'minmax', 'absmax'之一。
        :param scaler_path: 保存或加载Scaler的文件路径 (e.g., 'saved/scalers/pulse_scaler.joblib')。
        :param shuffle: 是否打乱数据。在非训练模式下会被强制设为 False。
        :param validation_split: 验证集比例。在非训练模式下会被强制设为 0。
        :param num_workers: 数据加载的工作线程数。
        :param training: 是否为训练模式。这会影响Scaler的加载/保存行为。
        """
        if not os.path.exists(params_npz_path):
            raise FileNotFoundError(f"工况参数文件未找到: {params_npz_path}。")
        if not os.path.exists(processed_pulses_path):
             raise FileNotFoundError(f"预处理波形文件未找到: {processed_pulses_path}。")

        target_scaler = None
        # --- Scaler 处理 ---
        if pulse_norm_mode != 'none':
            if scaler_path is None:
                raise ValueError("当 pulse_norm_mode 不为 'none' 时, 必须提供 'scaler_path'。")

            # 如果是训练模式，则拟合并保存Scaler
            if training:
                print(f"训练模式：正在为目标波形拟合Scaler (模式: {pulse_norm_mode})...")
                
                with np.load(processed_pulses_path) as data:
                    all_waveforms_data = [data[key] for key in data.keys()]
                
                if not all_waveforms_data: 
                    raise ValueError(f"未能从 {processed_pulses_path} 加载任何波形数据。")
                
                full_dataset_np = np.concatenate(all_waveforms_data, axis=0)

                if pulse_norm_mode == 'minmax':
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    target_scaler = scaler.fit(full_dataset_np.reshape(-1, 1))
                    print(f"MinMaxScaler 拟合完毕。Min: {target_scaler.data_min_[0]:.4f}, Max: {target_scaler.data_max_[0]:.4f}")
                
                elif pulse_norm_mode == 'absmax':
                    target_scaler = PulseAbsMaxScaler().fit(full_dataset_np)
                    print(f"PulseAbsMaxScaler 拟合完毕。全局绝对值Max: {target_scaler.data_abs_max_:.4f}")
                else:
                    raise ValueError(f"未知的 pulse_norm_mode: {pulse_norm_mode}")
                
                # 创建目录并保存Scaler
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                joblib.dump(target_scaler, scaler_path)
                print(f"Scaler 已保存至: {scaler_path}")

            # 如果是测试模式，则直接加载Scaler
            else:
                print(f"测试模式：正在从 {scaler_path} 加载Scaler...")
                try:
                    target_scaler = joblib.load(scaler_path)
                    print("Scaler 加载成功。")
                except FileNotFoundError:
                    raise FileNotFoundError(f"Scaler文件未找到: {scaler_path}。请先在训练模式下运行以生成Scaler文件。")
        
        # --- 根据模式调整参数 ---
        if not training:
            shuffle = False
            validation_split = 0.0

        # --- 实例化Dataset ---
        self.dataset = CollisionDataset(params_npz_path, processed_pulses_path, target_scaler)
        
        # 保存一些有用的属性
        self.pulse_norm_mode = pulse_norm_mode
        self.training = training
        self.target_scaler = target_scaler

        # --- 调用父类构造函数 ---
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)