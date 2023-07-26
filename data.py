from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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

class TestSet(Dataset):  # 图片预处理
    def __init__(self, img_path):
        self.img_path = img_path
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  # 数据调整到指定大小，并进行归一化

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = self.img_path[idx]
        img = Image.open(img).convert("RGB")
        img = self.transform(img)
        return img