import copy

import torch
import torchvision
import torchvision.transforms as transform
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from delta_compressor import qd_compressor
from quantizator import quantize_model
import time
# viz = visdom.Visdom()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception,self).__init__()
        #1*1卷积
        self.branch1 = nn.Sequential(nn.Conv2d(in_planes,n1x1,kernel_size=1),
                                     nn.BatchNorm2d(n1x1),nn.ReLU(True)
                                     )
        #1*1和3*3卷积
        self.branch2 = nn.Sequential(nn.Conv2d(in_planes,n3x3red,kernel_size=1),
                                  nn.BatchNorm2d(n3x3red),nn.ReLU(True),
                                  nn.Conv2d(n3x3red,n3x3,kernel_size=3,padding=1),
                                  nn.BatchNorm2d(n3x3),nn.ReLU(True)
                                  )

        #1*1和2个3*3，相当于是1*1和1个5*5
        self.branch3 = nn.Sequential(nn.Conv2d(in_planes,n5x5red,kernel_size=1),
                                   nn.BatchNorm2d(n5x5red),nn.ReLU(True),
                                   nn.Conv2d(n5x5red,n5x5,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(n5x5),nn.ReLU(True),
                                   nn.Conv2d(n5x5,n5x5,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(n5x5),nn.ReLU(True)
        )

        #3*3池化和1*1
        self.branch4 = nn.Sequential(nn.MaxPool2d(3,stride=1,padding=1),
                                   nn.Conv2d(in_planes,pool_planes,kernel_size=1),
                                   nn.BatchNorm2d(pool_planes),nn.ReLU(True)
                                   )

    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1,out2,out3,out4],dim=1)

class GoogleNet(nn.Module):
    def __init__(self,out_dim):
        super(GoogleNet,self).__init__()
        self.pre_layer = nn.Sequential(nn.Conv2d(3,192,kernel_size=3,padding=1),
                                       nn.BatchNorm2d(192),nn.ReLU(True)
                                       )
        self.a3 = Inception(192,64,96,128,16,32,32)
        self.b3 = Inception(256,128,128,192,32,96,64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.a4 = Inception(480,192,96,208,16,48,64)
        self.b4 = Inception(512,160,112,224,24,64,64)
        self.c4 = Inception(512,128,128,256,24,64,64)
        self.d4 = Inception(512,112,144,288,32,64,64)
        self.e4 = Inception(528,256,160,320,32,128,128)
        self.a5 = Inception(832,256,160,320,32,128,128)
        self.b5 = Inception(832,384,192,384,48,128,128)
        self.avgpool = nn.AvgPool2d(kernel_size=16,stride=1)
        self.linear = nn.Linear(1024,out_dim)

    def forward(self,x):
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
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        return x



if __name__ == '__main__':
    model = GoogleNet(out_dim=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    loss_list,acc_list=[],[]
    quantized_model_old = copy.deepcopy(model)
    quantized_model_new = copy.deepcopy(model)
    # 添加tensorboard画图可视化
    # writer = SummaryWriter("./result/resnet18/resnet18_origin_model")
    total_train_loss = 0
    total_train_step = 0
    total_train_time = 0
    total_compress_time = 0
    total_quantize_time = 0
    for epoch in range(5):
        if epoch != 0:
            quantized_model_old_state_dict = copy.deepcopy(quantized_model_new_state_dict)
            quantized_model_old.load_state_dict(quantized_model_old_state_dict)
        correct = 0
        total = 0
        train_time_t0 = time.time()
        for i,(inputs,labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #优化器梯度归零
            optimizer.zero_grad()
            #正向传播+反向传播+Adam优化
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_step += 1
            # writer.add_scalar("train_loss", loss.item(), total_train_step)
            if i % 100 == 0:
                print("Epoch:%d,Minibatch:%d,loss:%.4f"%(epoch+1,i+1,loss.item()))
                print('Accuracy of the network on the train images: %d %%' % (
                        100 * correct / total))
        train_time_t1 = time.time()
        total_train_time += (train_time_t1 - train_time_t0)
        loss_list.append(loss.item())
        acc_list.append(correct / total)
        torch.save(model, f"./model/googlenet/googlenet_lr001_epoch{epoch}.pth")
        quantize_time_t0 = time.time()
        S, Z, quantized_model_new_state_dict = quantize_model(model)
        quantize_time_t1 = time.time()
        total_quantize_time += (quantize_time_t1 - quantize_time_t0)
        quantized_model_new.load_state_dict(quantized_model_new_state_dict)
        if epoch != 0:
            compress_time_t0 = time.time()
            qd_compressor(quantized_model_old, quantized_model_new, S, Z,
                          path=f"./Snapshots/googlenet/googlenet_snapshot_epoch{epoch}",
                          compressed_s_path=f"./scales/googlenet/scale_epoch{epoch}",
                          compressed_z_path=f"./zero_points/googlenet/zero_point_epoch{epoch}")
            compress_time_t1 = time.time()
            total_compress_time += (compress_time_t1 - compress_time_t0)
    print("Train Finished!!!")
    loss_list,acc_list=[],[]
    correct = 0
    total = 0
    for i,(inputs,labels) in enumerate(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        #优化器梯度归零
        optimizer.zero_grad()
        #正向传播+反向传播+Adam优化
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Test Minibatch:%d,loss:%.4f"%(i+1,loss.item()))
            print('Accuracy of the network on the test images: %d %%' % (
                    100 * correct / total))
            loss_list.append(loss.item())
            acc_list.append(correct / total)
    print("Finished!!!")
    print(total_quantize_time / 50, total_compress_time / 50)
    print(total_train_time / 5)