# plot_accuracy_distribution.py

import warnings
warnings.filterwarnings('ignore')
import os
import json
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# 导入项目模块
import model.model as module_arch
from model.metric import ISORating
from utils.util import InputScaler, inverse_transform
from parse_config import ConfigParser # 仅用于日志记录器

#==========================================================================================
# 1. 配置文件 (请在此处修改为您本地的路径和配置)
#==========================================================================================
# 绘图中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# 1.1. 文件路径配置
# --------------------------------------------------------------------------------------
# 指定要加载的模型检查点 (.pth) 文件路径
CHECKPOINT_PATH = (
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\LX-model-PulsePredict\saved\models\PulseCNN_GauNLL\1106_123135\model_best.pth"
)

# 指定要分析的数据集 (.npz) 文件路径 (例如，测试集或包含所有工况的完整数据集)
DATASET_NPZ_PATH = (
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关"
    r"\VCS波形数据集打包\acc_data_before1103_5817\packaged_data_test.npz"
)

# 1.2. 绘图轴配置
# --------------------------------------------------------------------------------------
# 定义散点图的 X 轴和 Y 轴分别对应哪个工况参数
# 索引: 0 = 速度 (velocity), 1 = 角度 (angle), 2 = 重叠率 (overlap)
X_AXIS_PARAM_INDEX = 2  # 例如：重叠率
Y_AXIS_PARAM_INDEX = 1  # 例如：角度

# 坐标轴标签
PARAM_NAMES = ["Velocity (km/h)", "Angle (deg)", "Overlap"]
X_AXIS_LABEL = PARAM_NAMES[X_AXIS_PARAM_INDEX]
Y_AXIS_LABEL = PARAM_NAMES[Y_AXIS_PARAM_INDEX]

# 1.3. 工况范围筛选配置 (类似于 test.py)
# --------------------------------------------------------------------------------------
# 定义要分析的特定工况范围
# 如果要分析 DATASET_NPZ_PATH 中的所有数据，请将 'conditions' 设置为空列表: []
SPECIFIC_CASE_CONFIG = {
    'description': "全部分析工况",
    'conditions': [
        # 示例 1: 仅分析角度绝对值 >= 15 度
        # {'param_index': 1, 'type': 'abs_range', 'range': [15, 60]},
        
        # 示例 2: 仅分析速度在 40 到 50 之间
        # {'param_index': 0, 'type': 'range', 'range': [40, 50]},
        
        # 示例 3: 小角度 & 大重叠率 (同 test.py)
        # {'param_index': 1, 'type': 'abs_range', 'range': [0, 15]},    # 条件1: 角度绝对值在[0, 15]度
        # {'param_index': 2, 'type': 'abs_range', 'range': [0.75, 1.0]}  # 条件2: 重叠率绝对值在[0.75, 1.0]
    ]
}

# 1.4. 其他配置
# --------------------------------------------------------------------------------------
# 用于推理的批量大小
BATCH_SIZE = 512

# ISO Rating 的颜色映射范围 (vmin, vmax)
ISO_RATING_RANGE_X = (0.5, 1.0)  # X 轴范围
ISO_RATING_RANGE_Y = (0.1, 0.9)  # Y 轴范围
ISO_RATING_RANGE_Z = (0, 0.6)  # Z 轴范围

#==========================================================================================
# 2. 辅助函数
#==========================================================================================

def get_run_root_and_config_path(checkpoint_path):
    """
    根据检查点路径推断出实验的根目录和config.json路径。
    (借鉴自 export_onnx_烈度.py)
    """
    cp_path = Path(checkpoint_path)
    # 假设 .pth 文件在 session 文件夹中 (如 resume_... 或 test_...)
    config_path = cp_path.parent / 'config.json'
    
    if config_path.exists():
        # .pth 在 session 文件夹中, config 也在
        run_root_dir = cp_path.parent
    else:
        # .pth 在 session 文件夹中, config 在上一级 (标准的 resume 场景)
        config_path = cp_path.parent.parent / 'config.json'
        if config_path.exists():
            run_root_dir = cp_path.parent.parent
        else:
            # .pth 可能就在 run_root_dir 中
            config_path = cp_path.parent / 'config.json'
            if config_path.exists():
                run_root_dir = cp_path.parent
            else:
                raise FileNotFoundError(
                    f"在 {cp_path.parent} 或其父目录中均未找到 'config.json'。"
                )
    
    return run_root_dir, config_path

def plot_iso_scatter(x_data, y_data, color_data, config, save_path, vmin, vmax):
    """
    绘制并保存散点图。
    (此版本接受 vmin 和 vmax 参数)
    """
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(
        x_data, 
        y_data, 
        c=color_data, 
        cmap='viridis',  # 使用 viridis 颜色映射
        alpha=0.75,
        s=100,
        vmin=vmin,  # 使用传入的 vmin
        vmax=vmax   # 使用传入的 vmax
    )
    
    # 更新 colorbar 标签以显示动态范围
    cbar = plt.colorbar(sc, label=f"ISO Rating (Range: ({vmin:.2f}, {vmax:.2f}))") 
    cbar.ax.tick_params(labelsize=16)  # 添加这行,控制colorbar刻度数字大小
    cbar.set_label(f"ISO Rating (Range: ({vmin:.2f}, {vmax:.2f}))", fontsize=16)  # 控制colorbar标签字号
    
    plt.xlabel(config['x_label'], fontsize=18)
    plt.ylabel(config['y_label'], fontsize=18)
    plt.title(config['title'], fontsize=16, pad=10)
    plt.tick_params(axis='both', labelsize=16)  #控制刻度数字大小
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 在标题下方添加工况范围描述
    # plt.figtext(0.5, 0.95, f"Data Filter: {config['subtitle']}", ha="center", fontsize=10, style='italic')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
#==========================================================================================
# 3. 主执行函数
#==========================================================================================

def main():
    # --- 0. 初始化和日志 ---
    # 使用 ConfigParser 仅为了获取其 logger
    try:
        dummy_config = ConfigParser({}, resume=CHECKPOINT_PATH, is_test_run=True)
        logger = dummy_config.get_logger('plot_scatter')
        # 我们不使用 dummy_config.save_dir, 而是自己构造
    except Exception:
        # 如果ConfigParser失败，使用基本日志
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('plot_scatter')
        logger.warning("未能初始化 ConfigParser，使用基本日志。")

    logger.info("开始执行精度分布散点图绘制脚本...")
    
    # --- 1. 确定路径和加载配置 ---
    checkpoint_path = Path(CHECKPOINT_PATH)
    try:
        run_root_dir, config_path = get_run_root_and_config_path(checkpoint_path)
    except FileNotFoundError as e:
        logger.error(e)
        return

    logger.info(f"加载模型检查点: {checkpoint_path}")
    logger.info(f"加载配置文件: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)

    # 创建保存目录
    save_plot_dir = run_root_dir / "scatter_plots"
    os.makedirs(save_plot_dir, exist_ok=True)
    logger.info(f"图表将保存至: {save_plot_dir}")

    # --- 2. 加载模型 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_arch_type = config['arch']['type']
    model_arch_args = config['arch']['args']
    
    try:
        model = getattr(module_arch, model_arch_type)(**model_arch_args).to(device)
    except Exception as e:
        logger.error(f"加载模型架构 '{model_arch_type}' 失败: {e}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    logger.info(f"模型 '{model_arch_type}' 加载成功并设置到 {device}。")

    # --- 3. 加载 Scalers ---
    input_scaler = InputScaler(v_min=23.5, v_max=65, a_abs_max=60, o_abs_max=1)
    target_scaler = None
    
    try:
        scaler_path_str = config['data_loader_train']['args'].get('scaler_path')
        if scaler_path_str:
            scaler_path = Path(scaler_path_str)
            if scaler_path.exists():
                target_scaler = joblib.load(scaler_path)
                logger.info(f"成功加载目标波形 Scaler: {scaler_path}")
            else:
                logger.warning(f"Scaler 文件 '{scaler_path}' 未找到。")
        else:
            logger.warning("配置中未指定 'scaler_path'。")
        
        if target_scaler is None:
             logger.warning("未加载 Target Scaler，将使用归一化尺度进行评估（可能不准确）。")
             
    except Exception as e:
        logger.error(f"加载 Scaler 时出错: {e}")
        return

    # --- 4. 加载并筛选数据 ---
    logger.info(f"正在加载并筛选数据集: {DATASET_NPZ_PATH}")
    try:
        data = np.load(DATASET_NPZ_PATH)
        all_raw_params = data['params']   # (N, 3)
        all_waveforms = data['waveforms'] # (N, 3, 150)
    except Exception as e:
        logger.error(f"加载 .npz 文件失败: {e}")
        return

    # 应用工况筛选
    conditions = SPECIFIC_CASE_CONFIG.get('conditions', [])
    if not conditions:
        logger.info("未定义筛选条件，将分析数据集中的所有样本。")
        filtered_indices = np.arange(all_raw_params.shape[0])
    else:
        logger.info(f"根据 {len(conditions)} 个条件筛选工况...")
        combined_mask = np.full(all_raw_params.shape[0], True)
        for cond in conditions:
            param_index = cond['param_index']
            cond_type = cond['type']
            min_val, max_val = cond['range']
            
            param_to_check = all_raw_params[:, param_index]
            
            if cond_type == 'abs_range':
                current_mask = (np.abs(param_to_check) >= min_val) & (np.abs(param_to_check) <= max_val)
            else: # 'range'
                current_mask = (param_to_check >= min_val) & (param_to_check <= max_val)
            
            combined_mask &= current_mask
        
        filtered_indices = np.where(combined_mask)[0]

    if len(filtered_indices) == 0:
        logger.error("筛选后无任何样本，脚本终止。")
        return

    logger.info(f"筛选完毕，共 {len(filtered_indices)} / {all_raw_params.shape[0]} 个样本待处理。")

    # 提取筛选后的数据
    filtered_raw_params = all_raw_params[filtered_indices]
    filtered_true_waveforms = all_waveforms[filtered_indices]

    # --- 5. 准备模型输入 (归一化) ---
    normalized_params_list = []
    for params in filtered_raw_params:
        norm_vel, norm_ang, norm_ov = input_scaler.transform(params[0], params[1], params[2])
        normalized_params_list.append([norm_vel, norm_ang, norm_ov])
    
    normalized_params_np = np.array(normalized_params_list, dtype=np.float32)
    
    # 转换为 PyTorch Tensors
    norm_params_tensor = torch.from_numpy(normalized_params_np)
    true_waveforms_tensor = torch.from_numpy(filtered_true_waveforms).float()

    # --- 6. 执行批量推理 ---
    logger.info("开始执行模型批量推理...")
    all_pred_waveforms_orig = []
    
    # 使用 DataLoader 自动处理批次
    inference_dataset = TensorDataset(norm_params_tensor, true_waveforms_tensor)
    inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    GauNll_use = model_arch_args.get('GauNll_use', True) # 默认GauNll
    
    with torch.no_grad():
        for batch_data, batch_target in tqdm(inference_loader, desc="Inference"):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device) # 目标也放上GPU，以便inverse_transform

            # ------------------- 模型推理 ----------------------
            output = model(batch_data)
            
            # 提取用于评估的输出 (通常是均值)
            if model_arch_type == 'PulseMLP':
                metrics_output = model.get_metrics_output(output)
            elif model_arch_type == 'PulseCNN':
                metrics_output = model.get_metrics_output(output)
            else:
                metrics_output = output[0][-1] if GauNll_use else output[-1]
            # ---------------------------------------------------

            # 逆变换到物理尺度
            pred_orig, _ = inverse_transform(metrics_output, batch_target, target_scaler)
            
            all_pred_waveforms_orig.append(pred_orig.cpu())

    # 拼接所有批次的预测结果
    all_pred_waveforms_orig = torch.cat(all_pred_waveforms_orig, dim=0) # (N_filtered, 3, 150)
    
    logger.info("模型推理完成。")

    # --- 7. 计算 ISO Ratings ---
    logger.info("正在计算 ISO Ratings (逐样本)...")
    iso_ratings_x = []
    iso_ratings_y = []
    iso_ratings_z = []

    # 逐样本计算 (ISORating 类目前不支持批量)
    for i in tqdm(range(len(filtered_indices)), desc="Calculating ISO"):
        pred_wave = all_pred_waveforms_orig[i].numpy() # (3, 150)
        true_wave = filtered_true_waveforms[i]         # (3, 150)
        
        # X
        iso_x = ISORating(analyzed_signal=pred_wave[0, :], reference_signal=true_wave[0, :]).calculate()
        iso_ratings_x.append(iso_x)
        
        # Y
        iso_y = ISORating(analyzed_signal=pred_wave[1, :], reference_signal=true_wave[1, :]).calculate()
        iso_ratings_y.append(iso_y)
        
        # Z
        iso_z = ISORating(analyzed_signal=pred_wave[2, :], reference_signal=true_wave[2, :]).calculate()
        iso_ratings_z.append(iso_z)

    logger.info("ISO Ratings 计算完成。")

    # --- 8. 准备绘图数据 ---
    plot_x_data = filtered_raw_params[:, X_AXIS_PARAM_INDEX]
    plot_y_data = filtered_raw_params[:, Y_AXIS_PARAM_INDEX]

# --- 9. 绘图并保存 ---
    filter_desc = SPECIFIC_CASE_CONFIG['description']
    
    # 绘制 X-Rating
    logger.info("正在绘制 X-Rating 散点图...")
    plot_config_x = {
        'x_label': X_AXIS_LABEL,
        'y_label': Y_AXIS_LABEL,
        'title': f"ISO Rating (X-Axis) vs. {Y_AXIS_LABEL} and {X_AXIS_LABEL}",
        'subtitle': filter_desc
    }
    save_path_x = save_plot_dir / "iso_scatter_X.png"
    # 传入 X 轴的特定范围
    plot_iso_scatter(
        plot_x_data, plot_y_data, iso_ratings_x, plot_config_x, save_path_x,
        vmin=ISO_RATING_RANGE_X[0], vmax=ISO_RATING_RANGE_X[1]
    )

    # 绘制 Y-Rating
    logger.info("正在绘制 Y-Rating 散点图...")
    plot_config_y = {
        'x_label': X_AXIS_LABEL,
        'y_label': Y_AXIS_LABEL,
        'title': f"ISO Rating (Y-Axis) vs. {Y_AXIS_LABEL} and {X_AXIS_LABEL}",
        'subtitle': filter_desc
    }
    save_path_y = save_plot_dir / "iso_scatter_Y.png"
    # 传入 Y 轴的特定范围
    plot_iso_scatter(
        plot_x_data, plot_y_data, iso_ratings_y, plot_config_y, save_path_y,
        vmin=ISO_RATING_RANGE_Y[0], vmax=ISO_RATING_RANGE_Y[1]
    )

    # 绘制 Z-Rating
    logger.info("正在绘制 Z-Rating 散点图...")
    plot_config_z = {
        'x_label': X_AXIS_LABEL,
        'y_label': Y_AXIS_LABEL,
        'title': f"ISO Rating (Z-Axis) vs. {Y_AXIS_LABEL} and {X_AXIS_LABEL}",
        'subtitle': filter_desc
    }
    save_path_z = save_plot_dir / "iso_scatter_Z.png"
    # 传入 Z 轴的特定范围
    plot_iso_scatter(
        plot_x_data, plot_y_data, iso_ratings_z, plot_config_z, save_path_z,
        vmin=ISO_RATING_RANGE_Z[0], vmax=ISO_RATING_RANGE_Z[1]
    )

    logger.info(f"所有绘图已完成并保存至: {save_plot_dir}")

if __name__ == "__main__":
    main()