# coding=utf-8
import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid

device='cpu'
batch_size = 128
num_epoch = 1
z_dimension = 2
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
# 定义判别器
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,2,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.encoder_fc=nn.Linear(16*7*7,z_dimension)
        self.decoder_fc = nn.Linear(z_dimension,16 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0],-1)
        code = self.encoder_fc(x)
        x = self.decoder_fc(code)
        x = x.view(x.shape[0],16,7,7)
        decode = self.decoder(x)
        return code,decode

# 创建AE对象
AE = autoencoder().to(device)
AE.load_state_dict(torch.load('./AE_z2.pth'))
# 是单目标二分类交叉熵函数
criterion = nn.BCELoss()
ae_optimizer = torch.optim.Adam(AE.parameters(), lr=0.0003)
pos = []
label = []
###########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (img, lab) in enumerate(dataloader):
        num_img = img.size(0)
        # view()函数作用把img变成[batch_size,channel_size,784]
        img = img.view(num_img,  1,28,28).to(device)  # 将图片展开为28*28=784
        code,decode = AE(img)  # 将真实图片放入判别器中
        pos.append(code)
        label.append(lab)
        if(i==100):
            break
pos = torch.cat(pos)
label = torch.cat(label)
for i in range(10):
    plt.scatter(
        pos[label==i][:,0].detach().numpy(),
        pos[label==i][:,1].detach().numpy(),
        alpha=0.2,
        label=i)
plt.legend()
plt.title('AE-MNIST')
plt.show()
# 生成一个
import numpy as np
def plot_numbers(rangex,rangey):
    numbers = []
    x = torch.arange(rangex[0],rangex[1],(rangex[1]-rangex[0])/15)
    y = torch.arange(rangey[0],rangey[1],(rangey[1]-rangey[0])/15)
    for xx in x:
        for yy in y:
            z = torch.Tensor([[xx, yy]])
            out = AE.decoder_fc(z)
            out = out.view(out.shape[0], 16, 7, 7)
            decode = AE.decoder(out)
            numbers.append(decode)
    numbers = torch.cat(numbers)
    img = make_grid(numbers,nrow=len(x),normalize=True).detach().numpy()
    plt.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')
    plt.show()
plot_numbers([-2,2],[-2,2])