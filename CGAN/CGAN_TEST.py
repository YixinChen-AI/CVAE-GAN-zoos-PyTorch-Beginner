# coding=utf-8
import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  make_grid
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import os
# GPU
batch_size=128
device = 'cpu'
z_dimension = 100
# 图形啊处理过程
img_transform = transforms.Compose([
    transforms.ToTensor(),
])
# mnist dataset mnist数据集下载
mnist = datasets.MNIST(
    root='./data/', train=True, transform=img_transform, download=True
)
# data loader 数据载入
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True
)
####### 定义生成器 Generator #####
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dimension+10, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 784),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间
        )

    def forward(self, x):
        x = self.gen(x)
        return x
# 创建对象
G = generator()
G = G.to(device)
G.load_state_dict(torch.load('./generator_CGAN_z100.pth'))
#########判别器训练train#####################
outputs = []
for num in range(10):
    label = torch.Tensor([num]).repeat(8).long()
    label_onehot = torch.zeros((8,10))
    label_onehot[torch.arange(8),label]=1
    z = torch.randn((8,z_dimension))
    z = torch.cat([z,label_onehot],1)
    print(z.shape)
    outputs.append(G(z).view(z.shape[0],1,28,28))
outputs = torch.cat(outputs)
img = make_grid(outputs,nrow=8,normalize=False).clamp(0,1).detach().numpy()
plt.imshow(np.transpose(img,(1,2,0)),interpolation='nearest')
plt.show()
