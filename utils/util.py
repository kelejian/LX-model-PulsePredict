import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def get_case_ids_from_directory(waveform_dir, axis=None):
    """
    从波形文件目录中提取 case_id 列表
    
    :param waveform_dir: 存放波形 CSV 文件的目录路径
    :param axis: 用于提取 case_id 的轴向文件前缀 ('x', 'y', 或 'z')
                 如果为 None，则获取同时拥有 x, y, z 三个方向波形文件的完整 case_id 列表
    :return: 排序后的 case_id 列表
    """
    import os
    import re
    
    if not os.path.exists(waveform_dir):
        raise FileNotFoundError(f"目录不存在: {waveform_dir}")
    
    # 如果 axis 为 None，获取完整的 case_id 列表
    if axis is None:
        # 分别获取三个轴向的 case_id
        x_case_ids = set()
        y_case_ids = set()
        z_case_ids = set()
        
        for axis_name, case_set in [('x', x_case_ids), ('y', y_case_ids), ('z', z_case_ids)]:
            pattern = rf'^{axis_name}(\d+)\.csv$'
            for filename in os.listdir(waveform_dir):
                match = re.match(pattern, filename)
                if match:
                    case_id = int(match.group(1))
                    case_set.add(case_id)
        
        # 取交集，确保每个 case_id 都有完整的三个文件
        complete_case_ids = sorted(list(x_case_ids & y_case_ids & z_case_ids))
        
        print(f"找到 {len(complete_case_ids)} 个完整的 case_id（同时拥有 x, y, z 文件）")
        
        # 检查缺失的文件
        all_case_ids = x_case_ids | y_case_ids | z_case_ids
        incomplete_case_ids = all_case_ids - set(complete_case_ids)
        
        if incomplete_case_ids:
            print(f"警告：以下 case_id 的文件不完整: {sorted(incomplete_case_ids)}")
        
        return complete_case_ids
    
    # 如果指定了特定轴向，获取该轴向的 case_id 列表
    else:
        if axis not in ['x', 'y', 'z']:
            raise ValueError(f"axis 参数必须是 'x', 'y', 'z' 中的一个，当前值: {axis}")
        
        case_ids = []
        pattern = rf'^{axis}(\d+)\.csv$'  # 匹配格式如 x10.csv, y46.csv 等
        
        # 遍历目录中的所有文件
        for filename in os.listdir(waveform_dir):
            match = re.match(pattern, filename)
            if match:
                case_id = int(match.group(1))  # 提取数字部分
                case_ids.append(case_id)
        
        # 排序并去重
        case_ids = sorted(list(set(case_ids)))
        
        print(f"在目录 {waveform_dir} 中找到 {len(case_ids)} 个 {axis} 轴的 case_id")
        return case_ids

def inverse_transform(pred_tensor, target_tensor, scaler):
    """
    对预测和目标张量进行逆变换，以还原到原始物理尺度。

    :param pred_tensor: 模型预测的归一化张量。
    :param target_tensor: 真实的归一化目标张量。
    :param scaler: 用于逆变换的scaler对象 (e.g., from scikit-learn)。
    :return: 包含逆变换后的 (pred_orig, target_orig) 的元组。
    """
    if scaler is None:
        return pred_tensor, target_tensor

    original_shape = pred_tensor.shape
    
    # 使用 .detach() 确保此操作不影响计算图
    pred_numpy = pred_tensor.detach().cpu().numpy().reshape(-1, 1)
    pred_orig = scaler.inverse_transform(pred_numpy).reshape(original_shape)
    pred_orig = torch.from_numpy(pred_orig).to(pred_tensor.device)

    target_numpy = target_tensor.detach().cpu().numpy().reshape(-1, 1)
    target_orig = scaler.inverse_transform(target_numpy).reshape(original_shape)
    target_orig = torch.from_numpy(target_orig).to(target_tensor.device)
    
    return pred_orig, target_orig

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
