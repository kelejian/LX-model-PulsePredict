# flake8: noqa
import torch
import torch.nn as nn
import torch.onnx
from torchinfo import summary
from torchviz import make_dot

# @profile
def test_model(
    model,
    inputs,
    labels,
    criterion=None,
    optimizer=None,
    onnx_file_path="model_test.onnx"
):
    """
    通用化模型测试函数：
    1. 接受任意模型实例化对象 `model`。
    2. 自定义输入 `inputs` 和标签 `labels`。
    3. 支持前向传播、反向传播、损失计算。
    4. 导出 ONNX 模型并验证。
    5. 输出模型详细信息。
    
    参数：
    - model: PyTorch 模型实例化对象: torch.nn.Module
    - inputs: 模型的输入张量: tensor 或 tulple(tensor1, tensor2, ...) 或 list(tensor1, tensor2, ...)
    - labels: 模型的真实标签张量（用于损失计算）: tensor
    - criterion: 损失函数实例化对象，默认为 nn.MSELoss
    - optimizer: 优化器实例化对象，默认为 Adam
    - onnx_file_path: 导出的 ONNX 文件路径
    """
    # 默认损失函数和优化器
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 将模型设为训练模式
    model.train()
    print("\n~~~~~~~~~~~~~~~~~~~ 🚀🚀 开始测试神经网络模型是否可以正常训练 🚀🚀 ~~~~~~~~~~~~~~~~~~~~")
    # 打印模型结构信息
    print("\n============== 模型结构信息 ==============")
    _input_data = tuple(inputs) if isinstance(inputs, (tuple, list)) else inputs
    
    try:
        summary(
            model,
            input_data=_input_data,
            col_names=["input_size", "output_size", "num_params"],
            depth=3,
            device="cuda" if next(model.parameters()).is_cuda else "cpu"
        )
    except Exception as e:
        print(f"⚠ torchinfo.summary 执行失败: {e}")
        print("继续执行其他测试...")
    
    # 前向传播与loss计算
    print("\n============== 前向传播 ==============")
    if isinstance(inputs, (tuple, list)):
        outputs = model(*inputs)
        # 一行打印模型各个输入input的形状
        print(f"✔ 模型各个输入的形状：{[input.shape for input in inputs]}")

    else:
        outputs = model(inputs)
        print(f"✔ 输入形状：{inputs.shape}")

    # 检查 outputs 是否为 tuple 或 list,但排除 torch.Tensor
    if isinstance(outputs, (tuple, list)) and not isinstance(outputs, torch.Tensor):
        # 一行打印模型各个输出output的形状
        try:
            output_info = []
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    output_info.append(f"Tensor{tuple(output.shape)}")
                elif isinstance(output, (tuple, list)):
                    # 处理嵌套的 tuple/list (如 GaussianNLL 输出)
                    sub_info = []
                    for sub_output in output:
                        if isinstance(sub_output, torch.Tensor):
                            sub_info.append(f"Tensor{tuple(sub_output.shape)}")
                        else:
                            sub_info.append(str(type(sub_output)))
                    output_info.append(f"({', '.join(sub_info)})")
                else:
                    output_info.append(str(type(output)))
            print(f"✔ 模型输出结构：[{', '.join(output_info)}]")
        except Exception as e:
            print(f"✔ 模型输出类型：{type(outputs)}, 包含 {len(outputs)} 个元素")
            print(f"✔ 各元素类型：{[type(output) for output in outputs]}")
        
        # 尝试从多尺度输出中找到匹配的输出进行 loss 计算
        loss = None
        matched_output = None
        
        for i, output in enumerate(outputs):
            # 处理 (mean, var) 元组格式
            if isinstance(output, (tuple, list)) and len(output) >= 1:
                current_output = output[0] if isinstance(output[0], torch.Tensor) else output
            else:
                current_output = output
            
            if isinstance(current_output, torch.Tensor):
                # 检查形状是否匹配
                if current_output.shape == labels.shape:
                    loss = criterion(current_output, labels)
                    matched_output = current_output
                    print(f"✔ 使用第 {i+1} 个输出（形状: {tuple(current_output.shape)}）计算损失: {loss.item():.6f}")
                    break
        
        if loss is None:
            # 如果没有完全匹配的，尝试使用最后一个输出
            print("⚠ 所有输出形状与标签不完全匹配，使用最后一个输出...")
            last_output = outputs[-1]
            if isinstance(last_output, (tuple, list)):
                last_output = last_output[0] if isinstance(last_output[0], torch.Tensor) else last_output
            
            if isinstance(last_output, torch.Tensor):
                # 尝试对齐标签形状
                if last_output.shape[-1] != labels.shape[-1]:
                    print(f"  调整标签形状: {tuple(labels.shape)} -> 插值到长度 {last_output.shape[-1]}")
                    aligned_labels = torch.nn.functional.interpolate(
                        labels, 
                        size=last_output.shape[-1], 
                        mode='linear', 
                        align_corners=False
                    )
                else:
                    aligned_labels = labels
                
                loss = criterion(last_output, aligned_labels)
                matched_output = last_output
                print(f"✔ 使用最后一个输出（形状: {tuple(last_output.shape)}）计算损失: {loss.item():.6f}")
            else:
                raise ValueError(f"无法从输出中提取有效的张量进行损失计算，最后输出类型: {type(last_output)}")

    else:
        print(f"✔ 模型输出形状：{outputs.shape}")
        if labels.shape == outputs.shape:
            loss = criterion(outputs, labels)
            matched_output = outputs
            print(f"✔ 损失值：{loss.item():.6f}")
        else: 
            print(f"⚠ 模型输出形状 {tuple(outputs.shape)} 与标签形状 {tuple(labels.shape)} 不匹配")
            # 尝试调整形状后计算损失
            try:
                if outputs.shape[-1] != labels.shape[-1]:
                    aligned_labels = torch.nn.functional.interpolate(
                        labels, 
                        size=outputs.shape[-1], 
                        mode='linear', 
                        align_corners=False
                    )
                    loss = criterion(outputs, aligned_labels)
                    matched_output = outputs
                    print(f"✔ 调整标签形状后计算损失值：{loss.item():.6f}")
                else:
                    loss = criterion(outputs.view_as(labels), labels)
                    matched_output = outputs
                    print(f"✔ 调整输出形状后计算损失值：{loss.item():.6f}")
            except Exception as e:
                print(f"✘ 无法通过调整形状计算损失: {e}")
                loss = criterion(outputs, labels)  # 强制计算以便后续反向传播测试
                matched_output = outputs


    # 反向传播
    print("\n============== 反向传播 ==============")
    try:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("✔ 反向传播正常~")
    except Exception as e:
        print(f"✘ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()

    # 可视化计算图
    print("\n============== 计算图可视化 ==============")
    try:
        graph = make_dot(loss, params=dict(model.named_parameters()))
        graph.render("model_computation_graph", format="png")
        print("✔ 计算图已保存为 'model_computation_graph.png'")
    except Exception as e:
        print(f"⚠ 计算图可视化失败: {e}")

    # 导出 ONNX 模型
    print("\n============== 导出 ONNX 模型 ==============")
    
    try:
        # 根据输入类型配置输入名称和动态轴
        if isinstance(inputs, (tuple, list)):
            input_names = [f"input_{i}" for i in range(len(inputs))]
            dynamic_axes = {f"input_{i}": {0: "batch_size"} for i in range(len(inputs))}
        else:
            input_names = ["input"]
            dynamic_axes = {"input": {0: "batch_size"}}
        
        # 配置输出名称和动态轴（处理多尺度和嵌套结构）
        output_names = []
        output_idx = 0
        
        if isinstance(outputs, (tuple, list)) and not isinstance(outputs, torch.Tensor):
            for stage_output in outputs:
                if isinstance(stage_output, (tuple, list)):
                    for sub_output in stage_output:
                        if isinstance(sub_output, torch.Tensor):
                            output_names.append(f"output_{output_idx}")
                            dynamic_axes[f"output_{output_idx}"] = {0: "batch_size"}
                            output_idx += 1
                elif isinstance(stage_output, torch.Tensor):
                    output_names.append(f"output_{output_idx}")
                    dynamic_axes[f"output_{output_idx}"] = {0: "batch_size"}
                    output_idx += 1
        else:
            output_names = ["output"]
            dynamic_axes["output"] = {0: "batch_size"}
        
        torch.onnx.export(
            model,
            _input_data,
            onnx_file_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
        )
        print(f"✔ ONNX 模型已保存至 {onnx_file_path}")
        print("  在 https://netron.app/ 上查看 ONNX 模型结构")
    except Exception as e:
        print(f"⚠ ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from utils import read_json
    from parse_config import ConfigParser
    import model.model as module_arch

    # 模型实例化
    config = ConfigParser(read_json('config.json'))
    Pulsemodel = config.init_obj('arch', module_arch)
    
    # 将模型移动到CUDA设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Pulsemodel = Pulsemodel.to(device)

    # 示例输入数据（根据模型实际输入维度）
    batch_size = 128

    # 修正：模型输入应为 (B, input_dim)，默认 input_dim=3
    x = torch.randn(batch_size, 3).to(device)  # 工况特征输入
    
    # 标签数据（目标加速度波形）
    y = torch.randn(batch_size, 3, 150).to(device)  # (B, C, L) 三轴加速度

    print(f"\n{'='*80}")
    print(f"模型类型: {type(Pulsemodel).__name__}")
    print(f"输入数据形状: {x.shape}")
    print(f"标签数据形状: {y.shape}")
    print(f"计算设备: {device}")
    print(f"{'='*80}")

    # 测试模型
    test_model(Pulsemodel, inputs=x, labels=y)
