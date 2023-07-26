from glob import glob
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


# model = torch.load("vgg_lr002_epoch70.pth")
class ImgData(Dataset):  # 图片预处理
    def __init__(self, img_path, img_label):
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  # 数据调整到指定大小，并进行归一化

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = self.img_path[idx]
        img = Image.open(img).convert("RGB")
        img = self.transform(img)
        return img, self.img_label[idx]


def model_test(model, img_path, label_path):
    img_path = sorted(glob(img_path))
    img_path = img_path[:1000]
    matdata = loadmat(label_path)
    labels = matdata['gt_labels']
    labels = torch.tensor(labels[:1000]).reshape(-1).long() - 1
    dataset = ImgData(img_path, labels)
    data_iter = DataLoader(dataset, batch_size=64, shuffle=False)
    device = torch.device('cpu')
    model = model.to(device)
    length = len(dataset)
    # 验证步骤
    total_val_loss = 0
    total_accuracy = 0
    loss = nn.CrossEntropyLoss()
    print("start model test...")
    with torch.no_grad():
        for data in data_iter:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)
            val_loss = loss(output, targets)
            total_val_loss += val_loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            print(accuracy)
            total_accuracy += accuracy
    print("整体验证集上的loss:{}".format(total_val_loss))
    print("整体验证集上的正确率:{}".format(total_accuracy / length))

# img_path = '../data/train/*'
# label_path = '../data/train_labels.mat'
# model = torch.load("vgg19_lr005_epoch50_37.pth")
# print(model.state_dict())
# Model_Test(model, img_path, label_path)
#
# model = torch.load("vgg19_lr005_epoch50_3.pth")
# print(model.state_dict())
# Model_Test(model, img_path, label_path)
