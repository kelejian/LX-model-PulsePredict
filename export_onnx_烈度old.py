import yaml
import os
import copy
import json
import torch
# from thop import clever_format, profile
import numpy as np
import argparse
import model.model as module_arch
from parse_config import ConfigParser
import collections

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='LX-CrashPulsePredictionModel Training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)

    input_feature = torch.rand(1, 3)  # 加 batch 维度 (1, feature_dim) 速度、角度、重叠率
    onnx_path = r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\LX-model-PulsePredict\onnx_models\PulseCNN_no_dynamic.onnx"

    ## 模型参数
    model = config.init_obj('arch', module_arch)
    ckpt = torch.load(
        r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\LX-model-PulsePredict\saved\models\PulseCNN_GauNLL\1008_153448\resume_1008_161400\model_best.pth',
        map_location='cpu')  # 或 'cuda'
    state_dict = ckpt['state_dict']
    if ckpt['config']['arch'] != config['arch']:
        print("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
    model.load_state_dict(state_dict)
    model.eval()

    # calculate FLOPs, params
    # flops, params = profile(model, input_feature, verbose=False)
    # flops = flops * 2
    # flops, params = clever_format([flops, params], "%.3f")
    # print("Deploy Model ---> Total FLOPS: %s," % (flops), "Total params: %s" % (params))

    torch.onnx.export(
        model,                # 要导出的模型
        input_feature,        # 示例输入
        onnx_path,            # ONNX 文件的路径
        input_names=["input"],     # 输入名
        output_names = [
        "out_mean_s1", "out_var_s1",
        "out_mean_s2", "out_var_s2",
        "out_mean_s3", "out_var_s3"
        ],
         # out_mean2是最终输出的完整波形；前俩是粗糙短预测不用管。三个方差不用管
        do_constant_folding=True,
        opset_version=17,     # ONNX opset 版本
    )
    print("export onnx to:", onnx_path)
    import onnx
    from onnxsim import simplify

    print("start simplify onnx")
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print("onnx simplify done")

    # ============== ONNX Runtime 验证 ==============
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    onnx_inputs = {in_name: input_feature.cpu().numpy()}
    output_names = [o.name for o in sess.get_outputs()]
    onnx_outputs = sess.run(None, onnx_inputs)
    for name, out in zip(output_names, onnx_outputs):
        print(f"[{name}] shape={out.shape}")