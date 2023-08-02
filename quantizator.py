import math
import copy
import torch

# 量化函数
def quantize(value, scale, zero_point):
    # 将浮点数 value 根据 scale 和 zero_point 进行量化
    quantized_value = torch.round(value / scale + zero_point)
    quantized_value = quantized_value.to(torch.uint8)  # convert float32 to uint8
    return quantized_value


def dequantize(quantized_value, scale, zero_point, device='cuda:0'):
    # 将量化数值 quantized_value 还原为浮点数
    quantized_value = quantized_value.to(torch.float32)
    # if device == 'cpu':
    #     device = torch.device('cpu')
    # elif device == 'cuda:0':
    #     device = torch.device('cuda:0')
    # scale = scale.to(device)
    # zero_point = zero_point.to(device)
    quantized_value = quantized_value.to(device)
    dequantized_value = (quantized_value - zero_point) * scale
    return dequantized_value


def calculate_bit_width(model, L=1000, k=10):
    """
    This is a function which can calculate each layer's required bit-width of model by weighted entropy.

    Parameters:
     param1 - model
     param2 - this param decide the length of b list
     param3 - decide the division of parameter interval

    Returns:
     bit-width list
    """
    # 计算加权熵
    layer = 0
    E = [0] * L
    for param in model.parameters():
        weights = param.data
        weights = weights.reshape(-1)
        area_intersect = torch.histc(weights, bins=k, min=0, max=0)
        # 计算每层的加权熵
        n = weights.numel()
        for m in area_intersect:
            p = m / n
            if p != 0:
                E[layer] += p * math.log2(p)
        E[layer] *= -1
        layer += 1
    # 计算Emax,Emin
    Emin = float("inf")
    Emax = float("-inf")
    for i in range(layer):
        if Emin > E[i]:
            Emin = E[i]
        if Emax < E[i]:
            Emax = E[i]
    # 根据加权熵为每层分配位宽
    b = [0] * L
    bmin = 4
    bmax = 8
    for i in range(layer):
        b[i] = bmin + round(float((bmax - bmin) * ((E[i] - Emin) / (Emax - Emin))))
    return b


def calculate_s_and_z(model, b, L=1000):
    # 计算量化所需的S和Z
    S = [0] * L
    Z = [0] * L
    layer = 0
    for param in model.parameters():
        weights = param.data
        scale = (torch.max(weights) - torch.min(weights)) / (2 ** b[layer] - 1)
        if scale == 0:
            scale = 1e-8
        zero_point = 0 - torch.min(weights) / scale
        S[layer] = scale
        Z[layer] = zero_point
        layer += 1
    return S, Z


def quantize_model_and_save_state_dict(model, save_path):
    quantized_model = copy.deepcopy(model)
    b = calculate_bit_width(model)
    S, Z = calculate_s_and_z(model, b)
    for i, (orig_param, quant_param) in enumerate(zip(model.parameters(), quantized_model.parameters())):
        quant_param.data = quantize(orig_param.data, S[i], Z[i])
    torch.save(quantized_model.state_dict(), save_path)
    return S, Z


def deep_copy_model(model):
    model_copy = copy.deepcopy(model)
    for param in model_copy.parameters():
        param.requires_grad = False
    return model_copy


def quantize_model(model):
    """
    This is a function which can quantize model's parameters.

    Parameters:
     param1 - model

    Returns:
     scale, zero point and state dict of quantized model
    """
    quantized_model = deep_copy_model(model)
    b = calculate_bit_width(model)
    S, Z = calculate_s_and_z(model, b)
    for i, (orig_param, quant_param) in enumerate(zip(model.parameters(), quantized_model.parameters())):
        quant_param.data = quantize(orig_param.data, S[i], Z[i])
        # print(quant_param.data)
    return S, Z, quantized_model.state_dict()


def dequantize_model(load_model, dequantized_model, S, Z, device='cuda:0'):
    """
    This is a function which can dequantize quantized model.

    Parameters:
     param1 - model
     param2 - origin dequantized model,it must have same architecture with load model
     param3 - scale
     param4 - zero point
     param5 - work device:cpu or cuda:0

    Returns:
     none
    """
    for i, (orig_param, dequant_param) in enumerate(zip(load_model.parameters(), dequantized_model.parameters())):
        dequant_param.data = dequantize(orig_param.data, S[i], Z[i], device)
