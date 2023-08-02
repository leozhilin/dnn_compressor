import copy
import gzip
import pickle
import time

import torch
from torch import nn

import quantizator
from model_test import model_test, model_test_cifar10


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
        # print(delta_param.data)
    return delta


def delta_restore(modelx, delta):
    modely = copy.deepcopy(delta)
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
     param6 - save path of compressed scale
     param7 - save path of compressed zero point

    Returns:
     none
    """
    print("Compressing...")
    delta = delta_calculator(quantized_model_current, quantized_model_last)
    Compressor(delta.state_dict(), S, Z, compressed_file_name=path, compressed_s_path=compressed_s_path,
               compressed_z_path=compressed_z_path)


def qd_decompressor(current_model_path, current_version, restored_version, filename, device_name='cuda:0'):
    """
    This is a function which can quantize model's parameters.

    Parameters:
     param1 - current model path
     param2 - current model's version
     param3 - the version what you want to restore in
     param4 - model's name : vgg16, resnet18, googlenet and mobilenet
     param5 - device name : cuda:0 or cpu

    Returns:
     scale, zero point and state dict of quantized model
    """
    print("start qd decompressor")
    print("origin model path :", current_model_path)
    # 得到当前模型的量化版本
    if device_name == 'cuda:0':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        current_model = torch.load(current_model_path)
    elif device_name == 'cpu':
        device = torch.device('cpu')
        current_model = torch.load(current_model_path, map_location=torch.device('cpu'))
    quantized_current_model = copy.deepcopy(current_model)
    _, _, quantized_current_model_state_dict = quantizator.quantize_model(current_model)
    quantized_current_model.load_state_dict(quantized_current_model_state_dict)
    delta_model = copy.deepcopy(current_model)
    for i in range(restored_version - current_version): # restored_version > current_version
        print(f"current restore version: {current_version + i + 1}")
        # 载入delta文件
        S, Z, delta_model_state_dict = Decompressor(decompressed_file_name=f"./Snapshots/{filename}/{filename}_snapshot_epoch{current_version + i + 1}",
                                                    decompressed_s_path=f"./scales/{filename}/scale_epoch{current_version + i + 1}",
                                                    decompressed_z_path=f"./zero_points/{filename}/zero_point_epoch{current_version + i + 1}")
        delta_model.load_state_dict(delta_model_state_dict)
        # 差分恢复，得到恢复模型的量化版本
        restored_quantized_model = delta_restore(quantized_current_model, delta_model)
        quantized_current_model = restored_quantized_model
    restored_model = copy.deepcopy(current_model)
    quantizator.dequantize_model(restored_quantized_model, restored_model, S, Z, device=device_name)
    print("qd decompresor : finish !")
    return restored_model


if __name__ == '__main__':
    import torch.nn.functional as F
    class ResidualBlock(nn.Module):
        def __init__(self, inchannel, outchannel, stride=1):
            super(ResidualBlock, self).__init__()
            self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(outchannel)
                )

        def forward(self, x):
            out = self.left(x)
            out += self.shortcut(x)
            out = F.relu(out)
            return out


    class ResNet(nn.Module):
        def __init__(self, ResidualBlock, num_classes=10):
            super(ResNet, self).__init__()
            self.inchannel = 64
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
            self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
            self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
            self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
            self.fc = nn.Linear(512, num_classes)

        def make_layer(self, block, channels, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
            layers = []
            for stride in strides:
                layers.append(block(self.inchannel, channels, stride))
                self.inchannel = channels
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out


    def ResNet18():

        return ResNet(ResidualBlock)


    vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


    class VGG(nn.Module):
        def __init__(self, vgg):
            super(VGG, self).__init__()
            self.features = self._make_layers(vgg)
            self.dense = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
            )
            self.classifier = nn.Linear(4096, 10)

        def forward(self, x):
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.dense(out)
            out = self.classifier(out)
            return out

        def _make_layers(self, vgg):
            layers = []
            in_channels = 3
            for x in vgg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                    in_channels = x

            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
            return nn.Sequential(*layers)


    model = VGG(vgg)

    class Inception(nn.Module):
        def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
            super(Inception, self).__init__()
            # 1*1卷积
            self.branch1 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=1),
                                         nn.BatchNorm2d(n1x1), nn.ReLU(True)
                                         )
            # 1*1和3*3卷积
            self.branch2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=1),
                                         nn.BatchNorm2d(n3x3red), nn.ReLU(True),
                                         nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(n3x3), nn.ReLU(True)
                                         )

            # 1*1和2个3*3，相当于是1*1和1个5*5
            self.branch3 = nn.Sequential(nn.Conv2d(in_planes, n5x5red, kernel_size=1),
                                         nn.BatchNorm2d(n5x5red), nn.ReLU(True),
                                         nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(n5x5), nn.ReLU(True),
                                         nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(n5x5), nn.ReLU(True)
                                         )

            # 3*3池化和1*1
            self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                         nn.Conv2d(in_planes, pool_planes, kernel_size=1),
                                         nn.BatchNorm2d(pool_planes), nn.ReLU(True)
                                         )

        def forward(self, x):
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            out3 = self.branch3(x)
            out4 = self.branch4(x)
            return torch.cat([out1, out2, out3, out4], dim=1)


    class GoogleNet(nn.Module):
        def __init__(self, out_dim):
            super(GoogleNet, self).__init__()
            self.pre_layer = nn.Sequential(nn.Conv2d(3, 192, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(192), nn.ReLU(True)
                                           )
            self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
            self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
            self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
            self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
            self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
            self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
            self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
            self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
            self.avgpool = nn.AvgPool2d(kernel_size=16, stride=1)
            self.linear = nn.Linear(1024, out_dim)

        def forward(self, x):
            x = self.pre_layer(x)
            x = self.a3(x)
            x = self.b3(x)
            x = self.maxpool(x)
            x = self.a4(x)
            x = self.b4(x)
            x = self.c4(x)
            x = self.d4(x)
            x = self.e4(x)
            x = self.a5(x)
            x = self.b5(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            return x
    print(torch.cuda.is_available())

    # model = qd_decompressor("./model/googlenet/googlenet_lr001_epoch0.pth", 0, 1, "googlenet", "cuda:0")
    # model_test_cifar10(model)
    #
    # model1 = torch.load("./model/googlenet/googlenet_lr001_epoch1.pth")
    #
    # model_test_cifar10(model1)
    version = 49
    total_decompress_time = 0
    decompress_time_t0 = time.time()
    model = qd_decompressor("./model/shufflenet/shufflenet_lr001_epoch0.pth", 0, version, "shufflenet", "cuda:0")
    decompress_time_t1 = time.time()
    total_decompress_time = (decompress_time_t1 - decompress_time_t0) / (version - 0)
    print("total_decompress_time : ", total_decompress_time)
    model_test_cifar10(model)

    model1 = torch.load("./model/shufflenet/shufflenet_lr001_epoch49.pth")

    model_test_cifar10(model1)
