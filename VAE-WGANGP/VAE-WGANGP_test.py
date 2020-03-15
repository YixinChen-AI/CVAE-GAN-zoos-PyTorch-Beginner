from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import numpy as np

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.encoder_fc1=nn.Linear(32*7*7,nz)
        self.encoder_fc2=nn.Linear(32*7*7,nz)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(nz,32 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1,out2 = self.encoder(x),self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0],-1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0],-1))
        z = self.noise_reparameterize(mean,logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0],32,7,7)
        out3 = self.decoder(out3)
        return out3,mean,logstd

def plot_numbers(rangex, rangey):
    numbers = []
    x = torch.arange(rangex[0], rangex[1],(rangex[1]-rangex[0])/15)
    y = torch.arange(rangey[0], rangey[1], (rangey[1]-rangey[0])/15)
    for xx in x:
        for yy in y:
            z = torch.Tensor([[xx, yy]])
            out = vae.decoder_fc(z)
            out = out.view(out.shape[0], 32, 7, 7)
            decode = vae.decoder(out)
            numbers.append(decode)
    numbers = torch.cat(numbers)
    img = make_grid(numbers, nrow=len(x), normalize=True).detach().numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    dataset = 'cifar10'
    dataset = 'mnist'
    batchSize = 128
    imageSize = 28
    nz=2
    nepoch=1
    print("Random Seed: 88")
    random.seed(88)
    torch.manual_seed(88)
    device = 'cpu'
    # 可以优化运行效率
    cudnn.benchmark = True
    dataset = dset.MNIST(root='./data',
                         train=True,
                         transform=transforms.Compose([transforms.ToTensor()]),
                         download=True
                         )
    n_channel = 1
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchSize,
                                             shuffle=True)
    print("=====> 构建VAE")
    vae = VAE().to(device)
    vae.load_state_dict(torch.load('./VAE-WGANGP-VAE_v2.pth'))
    pos=[];label=[];
    for epoch in range(nepoch):
        for i, (data,lab) in enumerate(dataloader, 0):
            num_img = data.size(0)
            data = data.view(num_img, 1, 28, 28).to(device)  # 将图片展开为28*28=784
            x, mean, logstd = vae(data)  # 将真实图片放入判别器中
            pos.append(mean)
            label.append(lab)
            if (i == 100):
                break
    pos = torch.cat(pos)
    label = torch.cat(label)
    print(pos.shape)
    print(label.shape)
    for i in range(10):
        plt.scatter(
            pos[label == i][:, 0].detach().numpy(),
            pos[label == i][:, 1].detach().numpy(),
            alpha=0.5,
            label=i)
    plt.title('VAE-WGANGP-MNIST')
    plt.legend()
    plt.show()

    plot_numbers([-1, 1], [-1, 1])
    # test 2
    # 这个任务中生成随机的几个数字，潜在变量维度为100
    nz=100
    print("=====> 构建VAE")
    vae = VAE().to(device)
    vae.load_state_dict(torch.load('./VAE-WGANGP-VAE.pth'))
    for i, (data,lab) in enumerate(dataloader, 0):
        number1 = data[0].view(1,1,28,28)
        number2 = data[1].view(1,1,28,28)
        label1 = lab[0]
        label2 = lab[1]
        recon_number1,mean1,logstd1 = vae(number1)
        recon_number2,mean2,logstd2 = vae(number2)
        mean_step = (mean2-mean1)/5
        logstd_step = (logstd2-logstd1)/5
        list = [number1]
        for n in range(5):
            z = vae.noise_reparameterize(mean1+(n+1)*mean_step,
                                         logstd1+(n+1)*logstd_step)
            out3 = vae.decoder_fc(z)
            out3 = out3.view(out3.shape[0],32,7,7)
            output = vae.decoder(out3)
            list.append(output)
        list.append(number2)
        break
    photos = torch.cat(list)
    img = make_grid(photos, nrow=7, normalize=True).detach().numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
    plt.show()