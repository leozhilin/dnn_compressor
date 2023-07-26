import copy
import gzip
import pickle

import torch

import quantizator
from model_test import model_test


def Compressor(state_dict, S, Z, compressed_file_name, compressed_s_path, compressed_z_path):
    """
    This is a function which can compress state dict of model.

    Parameters:
     param1 - model's state dict
     param2 - Scale
     param3 - zero point
     param4 - saved path of compressed model's state dict
     param5 - saved path of scale
     param6 - saved path of zero point

    Returns:
     none
    """
    # 提取模型参数
    model_parameters = state_dict
    # 将模型参数转换为字节数据
    parameters_bytes = pickle.dumps(model_parameters)
    # 压缩模型参数
    compressed_parameters = gzip.compress(parameters_bytes)
    compressed_s = gzip.compress(pickle.dumps(S))
    compressed_z = gzip.compress(pickle.dumps(Z))
    # 保存压缩后的模型参数到文件
    with open(compressed_file_name, 'wb') as f:
        f.write(compressed_parameters)
    with open(compressed_s_path, 'wb') as f:
        f.write(compressed_s)
    with open(compressed_z_path, 'wb') as f:
        f.write(compressed_z)


def Decompressor(decompressed_file_name, decompressed_s_path, decompressed_z_path):
    """
    This is a function which can decompress QDelta file.

    Parameters:
     param1 - save path of model's state dict
     param2 - save path of Scale
     param3 - save path of zero point

    Returns:
     Scale, zero point and model's state dict
    """
    with open(decompressed_file_name, 'rb') as f:
        compressed_parameters = f.read()
    with open(decompressed_s_path, 'rb') as f:
        compressed_s = f.read()
    with open(decompressed_z_path, 'rb') as f:
        compressed_z = f.read()
    # 解压缩模型参数
    decompressed_parameters = gzip.decompress(compressed_parameters)
    decompressed_s = gzip.decompress(compressed_s)
    decompressed_z = gzip.decompress(compressed_z)
    # 将解压缩后的字节数据转换回模型参数
    model_parameters = pickle.loads(decompressed_parameters)
    S = pickle.loads(decompressed_s)
    Z = pickle.loads(decompressed_z)
    state_dict = model_parameters
    return S, Z, state_dict


def delta_calculator(modelx, modely):
    delta = copy.deepcopy(modelx)
    for i, (delta_param, paramx, paramy) in enumerate(
            zip(delta.parameters(), modelx.parameters(), modely.parameters())):
        delta_param.data = (paramx.data - paramy.data) % 2 ** 8
    return delta


def delta_restore(modelx, delta):
    modely = copy.deepcopy(modelx)
    for i, (delta_param, paramx, paramy) in enumerate(
            zip(delta.parameters(), modelx.parameters(), modely.parameters())):
        paramy.data = (paramx.data + delta_param.data) % 2 ** 8
    return modely


def qd_compressor(quantized_model_last, quantized_model_current, S, Z, path, compressed_s_path, compressed_z_path):
    """
    This is a function which can compress delta of neighbor-version models and save compressed file by path.

    Parameters:
     param1 - last quantized model
     param2 - current quantized model
     param3 - scale
     param4 - zero point
     param5 - save path of compressed file

    Returns:
     none
    """
    print("Compressing...")
    delta = delta_calculator(quantized_model_current, quantized_model_last, device='cuda:0')
    Compressor(delta.state_dict(), S, Z, compressed_file_name=path, compressed_s_path=compressed_s_path,
               compressed_z_path=compressed_z_path)


def qd_decompressor(current_model_path, restored_version, device_name='cuda:0'):
    """
    This is a function which can quantize model's parameters.

    Parameters:
     param1 - model

    Returns:
     scale, zero point and state dict of quantized model
    """
    print("start qd decompressor")
    print("origin model path :", current_model_path)
    # 得到当前模型的量化版本
    if device_name == 'cuda:0':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    elif device_name == 'cpu':
        device = torch.device('cpu')
    current_model = torch.load(current_model_path).to(device)
    quantized_current_model = copy.deepcopy(current_model)
    _, _, quantized_current_model_state_dict = quantizator.quantize_model(current_model)
    quantized_current_model.load_state_dict(quantized_current_model_state_dict)
    delta_model = copy.deepcopy(current_model)
    for i in range(restored_version):
        print(f"current restore version: {i}")
        # 载入delta文件
        S, Z, delta_model_state_dict = Decompressor(decompressed_file_name=f"./Snapshots/Snapshot_epoch{i}",
                                                    decompressed_s_path=f"./scales/scale_epoch{i}",
                                                    decompressed_z_path=f"./zero_points/zero_point_epoch{i}")
        delta_model.load_state_dict(delta_model_state_dict)
        # 差分恢复，得到恢复模型的量化版本
        restored_quantized_model = delta_restore(quantized_current_model, delta_model)
        quantized_current_model = restored_quantized_model
    restored_model = copy.deepcopy(current_model)
    quantizator.dequantize_model(restored_quantized_model, restored_model, S, Z, device=device_name)
    print("qd decompresor : finish !")
    return restored_model


if __name__ == '__main__':
    img_path = './data/train/*'
    label_path = './data/train_labels.mat'
    model = qd_decompressor(current_model_path="./model/origin_vgg19_lr005_epoch50.pth", restored_version=10, device_name='cpu')
    model_test(model, img_path, label_path)
