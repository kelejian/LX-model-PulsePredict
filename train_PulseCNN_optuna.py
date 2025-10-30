import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')

import optuna
import torch
import numpy as np
import joblib
import json

# 导入您项目中的相关模块
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils import MetricTracker, inverse_transform
from parse_config import ConfigParser # 用于加载基础配置

# 为保证可复现性，可以设置随机种子
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def run_one_epoch(model, loader, criterions, metric_ftns, device, optimizer=None):
    """
    执行一个精简的训练或验证周期。
    此函数仿照 Trainer 类的核心逻辑。
    """
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    # 从 data_loader 获取 scaler 用于逆变换
    scaler = getattr(loader, 'target_scaler', None)
    
    # 初始化指标追踪器
    loss_names_to_track = [type(item['instance']).__name__ for item in criterions]
    metrics_tracker = MetricTracker('loss', *loss_names_to_track, *[m.__name__ for m in metric_ftns])

    with torch.set_grad_enabled(is_train):
        for data, target, case_ids in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss, loss_components = model.compute_loss(output, target, criterions)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics_tracker.update('loss', loss.item())
            for loss_name, loss_val in loss_components.items():
                metrics_tracker.update(loss_name, loss_val)

            metrics_output = model.get_metrics_output(output)
            metrics_output_orig, target_orig = inverse_transform(metrics_output, target, scaler)

            for met in metric_ftns:
                metrics_tracker.update(met.__name__, met(metrics_output_orig, target_orig))
                
    return metrics_tracker.result()


def objective(trial):
    """
    定义Optuna的优化目标函数。
    """
    # --- 1. 定义超参数搜索空间 ---
    # 优化器参数
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    # 模型架构参数
    mlp_latent_dim = trial.suggest_categorical("mlp_latent_dim", [128, 256, 384])
    mlp_num_layers = trial.suggest_int("mlp_num_layers", 2, 4)
    projection_init_channels = trial.suggest_categorical("projection_init_channels", [16, 32, 48, 64])
    
    # 损失函数权重 (这是一个关键的调优部分)
    loss_weight_multi = trial.suggest_float("loss_weight_multi", 0.5, 1.5)
    loss_weight_corridor = trial.suggest_float("loss_weight_corridor", 0.5, 1.5)
    loss_weight_slope = trial.suggest_float("loss_weight_slope", 1.0, 2.0)
    loss_weight_phase = trial.suggest_float("loss_weight_phase", 1.0, 2.0)
    loss_weight_initial = trial.suggest_float("loss_weight_initial", 0.2, 0.8)
    loss_weight_terminal = trial.suggest_float("loss_weight_terminal", 0.1, 0.5)

    # --- 2. 加载和准备 ---
    # 使用基础配置文件，然后用试算参数覆盖
    base_config = ConfigParser(json.load(open('config_CNN.json')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集 (使用基础配置中的路径)
    data_loader = base_config.init_obj('data_loader_train', module_data)
    valid_data_loader = data_loader.split_validation()
    if valid_data_loader is None:
        raise ValueError("Optuna需要验证集进行评估，请在config.json中设置 'validation_split' > 0")

    # --- 3. 根据试算参数构建模型和优化器 ---
    model_args = base_config['arch']['args']
    model_args.update({
        "mlp_latent_dim": mlp_latent_dim,
        "mlp_num_layers": mlp_num_layers,
        "projection_init_channels": projection_init_channels
    })
    model = module_arch.PulseCNN(**model_args).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 重新构建损失函数列表
    loss_configs = base_config['loss']
    weights_from_trial = [
        loss_weight_multi, loss_weight_corridor, loss_weight_slope,
        loss_weight_phase, loss_weight_initial, loss_weight_terminal
    ]
    criterions = []
    for i, loss_spec in enumerate(loss_configs):
        loss_instance = getattr(module_loss, loss_spec['type'])(**loss_spec.get('args', {}))
        criterions.append({
            'instance': loss_instance,
            'weight': weights_from_trial[i],
            'channel_weights': loss_spec.get('channel_weights',  [1, 1, 1])
        })
    
    # 评估指标
    metrics = [getattr(module_metric, met) for met in base_config['metrics']]

    # --- 4. 训练与评估循环 ---
    # Optuna中通常使用较少的Epochs进行快速评估
    epochs_for_optuna = 40 
    
    best_iso_rating_x = -1.0

    for epoch in range(1, epochs_for_optuna + 1):
        train_log = run_one_epoch(model, data_loader, criterions, metrics, device, optimizer=optimizer)
        val_log = run_one_epoch(model, valid_data_loader, criterions, metrics, device, optimizer=None)

        print(f"Trial {trial.number}, Epoch {epoch}/{epochs_for_optuna} - val_loss: {val_log['loss']:.4f}, val_iso_rating_x: {val_log['iso_rating_x']:.4f}")

        # 更新本次试算的最佳指标
        current_iso_rating_x = val_log['iso_rating_x']
        if current_iso_rating_x > best_iso_rating_x:
            best_iso_rating_x = current_iso_rating_x

        # Optuna剪枝逻辑 (Pruning)
        # 报告中间值，让Optuna判断是否要提前终止不佳的试算
        trial.report(current_iso_rating_x, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 返回本次试算最终要优化的目标值
    return best_iso_rating_x


if __name__ == "__main__":
    # --- Optuna 研究(Study)配置 ---
    # 定义研究名称和数据库路径，以便断点续训
    study_name = "pulse_cnn_iso_optimization_v1"
    db_path = "sqlite:///./saved/optuna_study_pulse_cnn.db"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    storage = optuna.storages.RDBStorage(db_path)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"成功从数据库加载研究 '{study_name}'。")
    except KeyError:
        print(f"未找到研究 '{study_name}'，将创建新的研究。")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize"  # 我们的目标是最大化 iso_rating_x
        )

    # 运行优化
    try:
        study.optimize(objective, n_trials=100) # 设置总的试算次数
    except KeyboardInterrupt:
        print("用户中断了优化。")
    finally:
        print("\n" + "="*50)
        print("              优化结束              ")
        print("="*50)

    # --- 打印最佳试算结果 ---
    print("\n最佳试算:")
    trial = study.best_trial
    print(f"  - 目标值 (Best val_iso_rating_x): {trial.value:.6f}")
    print("  - 最佳超参数:")
    for key, value in trial.params.items():
        print(f"    - {key}: {value}")