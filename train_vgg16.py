# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import copy
import time

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from delta_compressor import qd_compressor
from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

from quantizator import quantize_model

train_data = torchvision.datasets.CIFAR10(root="./", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
# model = torchvision.models.vgg19(pretrained=False)
# vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
#
#
# class VGG(nn.Module):
#     def __init__(self, vgg):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(vgg)
#         self.dense = nn.Sequential(
#             nn.Linear(512, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.4),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.4),
#         )
#         self.classifier = nn.Linear(4096, 10)
#
#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.dense(out)
#         out = self.classifier(out)
#         return out
#
#     def _make_layers(self, vgg):
#         layers = []
#         in_channels = 3
#         for x in vgg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

model = torchvision.models.vgg16(pretrained=False)
# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
EPOCH = 5

total_quantize_time = 0
total_compress_time = 0
total_train_time = 0
torch.save(model, "./model/vgg16/origin_vgg16_lr001_epoch70.pth")
quantized_model_old = copy.deepcopy(model)
quantized_model_new = copy.deepcopy(model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
for epoch in range(EPOCH):
    if epoch != 0:
        quantized_model_old_state_dict = copy.deepcopy(quantized_model_new_state_dict)
        quantized_model_old.load_state_dict(quantized_model_old_state_dict)
    print("-------第 {} 轮训练开始-------".format(epoch+1))

    # 训练步骤开始
    train_time_t0 = time.time()
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
    train_time_t1 = time.time()
    total_train_time += (train_time_t1 - train_time_t0)
    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    total_test_step = total_test_step + 1
    torch.save(model, f"./model/vgg16/vgg16_lr001_epoch{epoch}.pth")

    quantize_time_t0 = time.time()
    S, Z, quantized_model_new_state_dict = quantize_model(model)
    quantize_time_t1 = time.time()
    total_quantize_time += (quantize_time_t1 - quantize_time_t0)
    quantized_model_new.load_state_dict(quantized_model_new_state_dict)
    if epoch != 0:
        compress_time_t0 = time.time()
        qd_compressor(quantized_model_old, quantized_model_new, S, Z,
                      path=f"./Snapshots/vgg16/vgg16_snapshot_epoch{epoch}",
                      compressed_s_path=f"./scales/vgg16/scale_epoch{epoch}",
                      compressed_z_path=f"./zero_points/vgg16/zero_point_epoch{epoch}")
        compress_time_t1 = time.time()
        total_compress_time += (compress_time_t1 - compress_time_t0)

    print("模型已保存")
print(total_quantize_time / EPOCH, total_compress_time / EPOCH)
print(total_train_time / EPOCH)

