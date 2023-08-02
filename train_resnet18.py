import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
import torchvision.transforms as transform

from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F

from delta_compressor import qd_compressor
from quantizator import quantize_model
import time


# 残差块
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
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
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


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我行差们能够手动输入命令行参数，就是让风格变得和Linux命令不多
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
# parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  # 恢复训练时的模型路径
# args = parser.parse_args()

# 超参数设置
EPOCH = 5  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.001  # 学习率

# print("开始加载CIFAR10数据集!")
# 准备数据集并预处理
device = torch.device("cuda:0")

transform_train = transform.Compose([
    transform.RandomCrop(32,padding=4),
    transform.RandomHorizontalFlip(),
    transform.ToTensor(),
    transform.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

transform_test = transform.Compose([
    transform.ToTensor(),
    transform.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])


trainset = torchvision.datasets.CIFAR10(root='./',train=True,download=True,transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./',train=False,download=True,transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True,num_workers=0)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print("CIFAR10数据集加载完毕!")

# print("开始ResNet网络模型初始化!")
# 模型定义-ResNet
# model = torchvision.models.resnet18(pretrained=False)
model = ResNet18()
model = model.to(device)


# 定义损失函数和优化方式
loss_fn = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
loss_fn=loss_fn.to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0

# 添加tensorboard画图可视化
writer = SummaryWriter("./result/resnet18/resnet18_origin_model")

# 训练
if __name__ == "__main__":
    best_acc = 85  # 2 初始化best test accuracy
    best_epoch=0
    total_compress_time = 0
    total_quantize_time = 0
    total_train_time = 0
    print("开始训练! Resnet-18! 冲!")  # 定义遍历数据集的次数
    quantized_model_old = copy.deepcopy(model)
    quantized_model_new = copy.deepcopy(model)
    for epoch in range(pre_epoch, EPOCH):
        if epoch != 0:
            quantized_model_old_state_dict = copy.deepcopy(quantized_model_new_state_dict)
            quantized_model_old.load_state_dict(quantized_model_old_state_dict)
        print(f'--------第{epoch}轮训练开始---------')
        model.train()
        total_train_loss = 0.0
        correct = 0.0
        total = 0.0
        train_time_t0 = time.time()
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward + backward
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # 每训练100个batch打印一次loss和准确率
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total_train_step+=1
            correct += predicted.eq(labels.data).cpu().sum()
            if total_train_step % 100 == 0:
                print('[训练次数:%d] Loss: %.03f'% (total_train_step, total_train_loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        train_time_t1 = time.time()
        total_train_time += (train_time_t1 - train_time_t0)
        quantize_time_t0 = time.time()
        S, Z, quantized_model_new_state_dict = quantize_model(model)
        quantize_time_t1 = time.time()
        total_quantize_time += (quantize_time_t1 - quantize_time_t0)
        quantized_model_new.load_state_dict(quantized_model_new_state_dict)
        if epoch != 0:
            compress_time_t0 = time.time()
            qd_compressor(quantized_model_old, quantized_model_new, S, Z,
                          path=f"./Snapshots/resnet18/resnet18_snapshot_epoch{epoch}",
                          compressed_s_path=f"./scales/resnet18/scale_epoch{epoch}",
                          compressed_z_path=f"./zero_points/resnet18/zero_point_epoch{epoch}")
            compress_time_t1 = time.time()
            total_compress_time += (compress_time_t1 - compress_time_t0)
        torch.save(model, f"./model/resnet18/resnet18_lr001_epoch{epoch}.pth")
        print("开始测试!")
        with torch.no_grad():
            correct = 0
            total = 0
            total_test_loss=0
            for data in testloader:
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_test_loss+=loss.item()
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            print(f"测试集上的loss：{total_test_loss}")
            print(f'测试分类准确率为：{acc}')
            writer.add_scalar("test_loss", total_test_loss, total_test_step)
            writer.add_scalar("test_accuracy", acc, total_test_step)
            total_test_step = total_test_step + 1
            if acc > best_acc:
                best_acc = acc
    print(best_acc)
    print("训练结束!")
    print(total_quantize_time/EPOCH, total_compress_time/EPOCH)
    print(total_train_time / EPOCH)
