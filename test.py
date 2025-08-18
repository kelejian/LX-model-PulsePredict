import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import inverse_transform

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_test', module_data)

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

    scaler = data_loader.target_scaler

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
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
