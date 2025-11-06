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
    summary(
        model,
        input_data=_input_data,
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
        device="cuda" if next(model.parameters()).is_cuda else "cpu"
    )
    
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
            print(f"✔ 模型各个输出的形状：{[output.shape if isinstance(output, torch.Tensor) else type(output) for output in outputs]}")
        except Exception as e:
            print(f"✔ 模型输出类型：{type(outputs)}, 包含 {len(outputs)} 个元素")
            print(f"✔ 各元素类型：{[type(output) for output in outputs]}")
        
        loss = None
        for i, output in enumerate(outputs):
            if isinstance(output, torch.Tensor) and labels.shape == output.shape:
                loss = criterion(output, labels)
                print(f"✔ 第{i+1}个模型输出对应了一个loss值: {loss.item()}")
        
        if loss is None:
            print("✘ 所有模型输出形状与标签形状都不匹配，使用第一个输出计算损失")
            first_tensor = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs[0][0]
            loss = criterion(first_tensor, labels)
            print(f"✔ 损失值：{loss.item()}")

    else:
        print(f"✔ 模型输出形状：{outputs.shape}")
        if labels.shape == outputs.shape:
            loss = criterion(outputs, labels)
            print(f"✔ 损失值：{loss.item()}")
        else: 
            print("✘ 模型输出形状与标签形状不匹配，无法计算损失值")
            # 尝试调整形状后计算损失
            try:
                loss = criterion(outputs.view_as(labels), labels)
                print(f"✔ 调整形状后计算损失值：{loss.item()}")
            except:
                print("✘ 无法通过调整形状计算损失，请检查模型输出和标签的维度")
                loss = criterion(outputs, labels)  # 强制计算以便后续反向传播测试


    # 反向传播
    print("\n============== 反向传播 ==============")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("✔ 反向传播正常~")

    # 可视化计算图
    print("\n============== 计算图可视化 ==============")
    graph = make_dot(loss, params=dict(model.named_parameters()))
    graph.render("model_computation_graph", format="png")
    print("✔ 计算图已保存为 'model_computation_graph.png'")

    # 导出 ONNX 模型
    print("\n============== 导出 ONNX 模型 ==============")
    
    # 根据输入类型配置输入名称和动态轴
    if isinstance(inputs, (tuple, list)):
        input_names = [f"input_{i}" for i in range(len(inputs))]
        dynamic_axes = {f"input_{i}": {0: "batch_size"} for i in range(len(inputs))}
    else:
        input_names = ["input"]
        dynamic_axes = {"input": {0: "batch_size"}}
    
    # 配置输出名称和动态轴
    if isinstance(outputs, (tuple, list)) and not isinstance(outputs, torch.Tensor):
        output_names = [f"output_{i}" for i in range(len(outputs))]
        for i in range(len(outputs)):
            dynamic_axes[f"output_{i}"] = {0: "batch_size"}
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
    print("在 https://netron.app/ 上查看 ONNX 模型结构")

    # # 使用 ONNX Runtime 推理
    # print("\n============== ONNX Runtime 推理 ==============")
    # ort_session = onnxruntime.InferenceSession(onnx_file_path)
    # ort_inputs = {
    #     onnx_model.graph.input[i].name: (
    #         inputs[i].cpu().numpy() if isinstance(inputs, (tuple, list))
    #         else inputs.cpu().numpy()
    #     )
    #     for i in range(len(onnx_model.graph.input))
    # }
    # ort_outs = ort_session.run(None, ort_inputs)
    # print(f"ONNX 推理输出：{ort_outs}")

if __name__ == "__main__":
    from utils import read_json
    from parse_config import ConfigParser
    import model.model as module_arch

    # 模型实例化
    config = ConfigParser(read_json('config_CNN.json'))
    Pulsemodel = config.init_obj('arch', module_arch)
    
    # 将模型移动到CUDA设备
    Pulsemodel = Pulsemodel.cuda()

    # 示例输入数据（模拟数据集第1个batch）
    batch_size = 128

    y = torch.randn(batch_size, 3, 150).cuda()
    x = torch.randn(batch_size, 3).cuda()  # 随机生成连续特征

    # 测试模型
    test_model(Pulsemodel, inputs=x, labels=y)
