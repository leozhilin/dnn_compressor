from data import *
from glob import glob
from scipy.io import loadmat
import torchvision
from torch import nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

import time

from delta_compressor import qd_compressor
from quantizator import *
import datetime
def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
t0 = time.time()


if __name__ == '__main__':
    img_path = './data/train/*'
    label_path = './data/train_labels.mat'
    test_img_path = './data/test/*'

    img_path = sorted(glob(img_path))
    train_img_path = img_path[:8000]  # 前8000作为训练集
    val_img_path = img_path[8000:]  # 取出后1000作为验证集
    test_img_path = glob(test_img_path)
    matdata = loadmat(label_path)
    labels = matdata['gt_labels']
    train_label = torch.tensor(labels[:8000]).reshape(-1).long() - 1
    val_label = torch.tensor(labels[8000:]).reshape(-1).long() - 1

    train_dataset = ImgData(train_img_path, train_label)
    val_dataset = ImgData(val_img_path, val_label)
    test_dataset = TestSet(test_img_path)

    len_val = len(val_dataset)

    train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_iter = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_iter = DataLoader(test_dataset, batch_size=64, shuffle=False)

    vgg = torchvision.models.vgg19(pretrained=False)
    vgg.add_module('add_linear', nn.Linear(1000, 3))
    #打印模型结构
    print(vgg)
    device = torch.device('cpu')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vgg.to(device)

    #writer = SummaryWriter("Object_Detection_lr002_epoch50_resnet")
    step = 0
    tensor_trans = transforms.ToTensor()
    _resize = transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC)
    optim = torch.optim.SGD(vgg.parameters(), lr=0.005)
    loss = nn.CrossEntropyLoss()

    epoch = 50
    total_training_step = 0

    torch.save(vgg, "./model/origin_vgg19_lr005_epoch50.pth")
    quantized_model_old = copy.deepcopy(vgg)
    quantized_model_new = copy.deepcopy(vgg)
    for i in range(epoch):
        if i == 0:
            S, Z, quantized_model_old_state_dict = quantize_model(vgg)
            quantized_model_old.load_state_dict(quantized_model_old_state_dict)
        else:
            quantized_model_old.load_state_dict(quantized_model_new_state_dict)
        print("----------第{}轮训练开始----------".format(i + 1))
        for data in train_iter:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            #计算输出
            output = vgg(imgs)

            # loss及优化器优化模型
            result_loss = loss(output, targets)
            optim.zero_grad()
            result_loss.backward()
            optim.step()
            total_training_step += 1

            # if total_training_step % 50 == 0:
            #writer.add_scalar("train_loss", result_loss.item(), total_training_step)
            print("训练次数:{}, loss:{}".format(total_training_step, result_loss.item()))
        S, Z, quantized_model_new_state_dict = quantize_model(vgg)
        quantized_model_new.load_state_dict(quantized_model_new_state_dict)
        qd_compressor(quantized_model_old, quantized_model_new, S, Z, path=f"./Snapshots/Snapshot_epoch{i}", compressed_s_path=f"./scales/scale_epoch{i}", compressed_z_path=f"./zero_points/zero_point_epoch{i}")
        #得到第i轮网络模型
        #验证步骤
        total_val_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in val_iter:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = vgg(imgs)
                val_loss = loss(output, targets)
                total_val_loss += val_loss.item()
                accuracy = (output.argmax(1) == targets).sum()
                total_accuracy += accuracy
        #writer.add_scalar("test_loss", total_val_loss, i)
        print("整体验证集上的loss:{}".format(total_val_loss))
        #writer.add_scalar("test_accuracy", total_accuracy/len_val, i)
        print("整体验证集上的正确率:{}".format(total_accuracy/len_val))
        torch.save(vgg, f"./model/origin_vgg19_lr005_epoch50_{i}.pth")


    #writer.close()
    t1 = time.time()
    training_time = t1 - t0


    training_time = format_time(training_time)
    print("总共训练时间为:{}".format(training_time))
