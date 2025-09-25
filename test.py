import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import inverse_transform, plot_waveform_comparison, InputScaler
from pathlib import Path
import numpy as np

def main(config):
    logger = config.get_logger('test')

    # --- 1. 定义分组评估配置 ---
    grouping_config = {
        'param_name': 'velocity',  # 按哪个参数分组: 'velocity', 'angle', 'overlap'
        'param_index': 0,          # 参数在(N, 3)输入中的索引: 0=速度, 1=角度, 2=重叠率
        'ranges': {                # 定义区间的名字和 [min, max] 范围
            'low_speed': [23, 35],
            'mid_speed': [35, 48],
            'high_speed': [48, 65]
        }
    }
    logger.info(f"将根据参数 '{grouping_config['param_name']}' 的不同范围对测试结果进行分组统计。")
    # ---------------------------

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_test', module_data)
    # 打印测试集数据量
    logger.info(f"测试集数据量: {len(data_loader.dataset)}")

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    # loss_fn = config.init_obj('loss', module_loss) # <-- 旧的初始化方式
    
    # +++ 新的初始化方式 +++
    loss_fns = []
    if 'loss' in config.config and isinstance(config['loss'], list):
        for loss_spec in config['loss']:
            loss_module_name = loss_spec['type']
            loss_module_args = loss_spec.get('args', {})
            loss_instance = getattr(module_loss, loss_module_name)(**loss_module_args)
            loss_fns.append({
                'instance': loss_instance,
                'weight': loss_spec.get('weight', 1.0),
                'channel_weights': loss_spec.get('channel_weights', [1.0, 1.0, 1.0])
            })
    else:
        # 兼容旧格式的配置文件
        criterion_instance = config.init_obj('loss', module_loss)
        loss_fns.append({'instance': criterion_instance, 'weight': 1.0})

    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # --- 2. 收集所有测试样本的结果 ---
    all_raw_params = []
    all_preds_orig = []
    all_targets_orig = []
    all_losses = []

    # 获取用于逆变换的 scaler
    target_scaler = getattr(data_loader, 'target_scaler', None) # 获取数据集中的scaler属性，如果有的话

    with torch.no_grad():
        for batch_idx, (data, target, case_ids) in enumerate(tqdm(data_loader, desc="Predicting")):
            data, target = data.to(device), target.to(device)
            # ------------------- 使用统一的模型接口 ----------------------
            output = model(data)
            loss, loss_components = model.compute_loss(output, target, loss_fns)
            metrics_output = model.get_metrics_output(output)
            # ------------------------------------------------------------
            
            # 收集每个样本的损失值；注意，损失函数reduction不是'none'，这里loss.item()是批次的平均损失；为简化，我们直接用批次平均损失代表该批次中每个样本的损失
            all_losses.extend([loss.item()] * data.shape[0])

            # 计算指标前，先对数据进行逆变换，如果没有scaler，则返回原始张量
            pred_mean_orig, target_orig = inverse_transform(metrics_output, target, target_scaler)
            
            # 收集逆变换后的工况、预测和目标
            input_scaler = getattr(data_loader.dataset, 'input_scaler', InputScaler(v_min=23.5, v_max=65, a_abs_max=60, o_abs_max=1))
            for i in range(data.shape[0]):
                norm_vel, norm_ang, norm_ov = data[i].cpu().numpy()
                raw_vel, raw_ang, raw_ov = input_scaler.inverse_transform(norm_vel, norm_ang, norm_ov)
                all_raw_params.append([raw_vel, raw_ang, raw_ov])
            
            all_preds_orig.append(pred_mean_orig.cpu())
            all_targets_orig.append(target_orig.cpu())

            # ------------------------------画图----------------------------------
            if batch_idx == 2:
                plot_samples(data, batch_idx, pred_mean_orig, target_orig, case_ids, input_scaler, config, logger)
            # --------------------------------------------------------------------

    # 将列表中的批次拼接成一个大的张量/数组
    all_preds_orig = torch.cat(all_preds_orig, dim=0)
    all_targets_orig = torch.cat(all_targets_orig, dim=0)
    all_raw_params = np.array(all_raw_params)
    all_losses = np.array(all_losses)

    # --- 3. 对全量和分组数据进行评估 ---
    logger.info("\n" + "="*50)
    logger.info(" 全量测试集评估结果 ".center(50, "="))
    logger.info("="*50)
    evaluate_subset(all_preds_orig, all_targets_orig, all_losses, metric_fns, logger)

    param_to_check = all_raw_params[:, grouping_config['param_index']]
    
    for range_name, (min_val, max_val) in grouping_config['ranges'].items():
        title = f" 分组评估: {range_name} ({grouping_config['param_name']}: [{min_val}, {max_val}]) "
        logger.info("\n" + "="*50)
        logger.info(title.center(50, "="))
        logger.info("="*50)
        
        # 找到在此区间的样本索引
        indices = np.where((param_to_check >= min_val) & (param_to_check <= max_val))[0]
        
        if len(indices) == 0:
            logger.info("该区间内无测试样本。")
            continue
            
        # 根据索引筛选子集
        subset_preds = all_preds_orig[indices]
        subset_targets = all_targets_orig[indices]
        subset_losses = all_losses[indices]
        
        evaluate_subset(subset_preds, subset_targets, subset_losses, metric_fns, logger, f"样本数: {len(indices)}")

def plot_samples(data, batch_idx, pred_mean_orig, target_orig, case_ids, input_scaler, config, logger):
    """
    为第一批样本绘图。
    """
    # num_samples_to_plot = min(20, data.shape[0])
    num_samples_to_plot = data.shape[0]
    print("\nPlotting first batch samples...")
    for j in range(num_samples_to_plot):
        # --- 从归一化输入中反算出原始工况参数 ---
        normalized_params = data[j].cpu().numpy()
        norm_vel, norm_ang, norm_ov = normalized_params[0], normalized_params[1], normalized_params[2]
        
        raw_vel, raw_ang, raw_ov = input_scaler.inverse_transform(norm_vel, norm_ang, norm_ov)

        collision_params = {'vel': raw_vel, 'ang': raw_ang, 'ov': raw_ov}
        
        pred_sample = pred_mean_orig[j]
        target_sample = target_orig[j]
        sample_case_id = case_ids[j].item()
        
        # 使用被重定向后的 config.save_dir
        plot_waveform_comparison(
            pred_wave=pred_sample,
            true_wave=target_sample,
            params=collision_params,
            case_id=sample_case_id,
            epoch='test',
            batch_idx=batch_idx,
            sample_idx=j,
            save_dir=config.save_dir
        )
    logger.info(f"\n绘图结果已保存至 '{config.save_dir}' 目录下的 'fig' 子目录中。\n")

def evaluate_subset(preds, targets, losses, metric_fns, logger, header_info=None):
    """
    计算并打印给定数据子集的各项指标。
    """
    if header_info:
        logger.info(header_info)
        
    log = {'loss': np.mean(losses)}
    for met in metric_fns:
        log[met.__name__] = met(preds, targets)
    
    # 格式化输出
    for key, value in log.items():
        logger.info('    {:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='LX-CrashPulsePredictionModel Test')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)