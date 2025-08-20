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
from utils import inverse_transform, plot_waveform_comparison

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_test', module_data)
    # 打印测试集数据量
    logger.info(f"测试集数据量: {len(data_loader.dataset)}")

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    loss_fn = config.init_obj('loss', module_loss)
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

    total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))
    total_metrics = {met.__name__: 0.0 for met in metric_fns} # 使用字典存储指标

    scaler = getattr(data_loader, 'target_scaler', None) # 获取数据集中的scaler属性，如果有的话

    with torch.no_grad():
        for i, (data, target, case_ids) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            # -----------------------ADJUST-------------------------------
            # output = model(data)
            # loss = loss_fn(output, target)
            # (在归一化尺度上)计算loss
            pred_mean, pred_var = model(data)
            loss = loss_fn(pred_mean, target, pred_var)
            # ------------------------------------------------------------
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            # 计算指标前，先对数据进行逆变换，如果没有scaler，则返回原始张量
            pred_mean_orig, target_orig = inverse_transform(pred_mean, target, scaler)
            # 在原始尺度上计算指标
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size
            for met in metric_fns:
                metric_name = met.__name__
                total_metrics[metric_name] += met(pred_mean_orig, target_orig) * batch_size
            # ------------------------------画图----------------------------------
            # 只对第一个批次 (i == 0) 进行绘图
            if i == 0:
                num_samples_to_plot = min(10, batch_size)
                
                for j in range(num_samples_to_plot):
                    # --- 从归一化输入中反算出原始工况参数 ---
                    normalized_params = data[j].cpu().numpy()
                    norm_vel, norm_ang, norm_ov = normalized_params[0], normalized_params[1], normalized_params[2]
                    
                    # 根据 data_loaders.py 中的归一化公式推出的反算公式
                    raw_vel = norm_vel * (65 - 25) + 25  # 速度反算
                    raw_ang = norm_ang * 60             # 角度反算
                    raw_ov = norm_ov * 1                # 重叠率反算

                    collision_params = {'vel': raw_vel, 'ang': raw_ang, 'ov': raw_ov}
                    # -----------------------------------------
                    
                    pred_sample = pred_mean_orig[j]
                    target_sample = target_orig[j]
                    sample_case_id = case_ids[j].item()
                    
                    plot_waveform_comparison(
                        pred_wave=pred_sample,
                        true_wave=target_sample,
                        params=collision_params,
                        case_id=sample_case_id,
                        epoch='test',
                        batch_idx=i,
                        sample_idx=j,
                        save_dir=config.save_dir
                    )
                logger.info(f"\nSome plots of first batch results saved to '{config.save_dir}' directory.\n")

            # --------------------------------------------------------------------

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    log.update({
        met_name: total_metrics[met_name] / n_samples for met_name in total_metrics
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='LX-Project Test')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
