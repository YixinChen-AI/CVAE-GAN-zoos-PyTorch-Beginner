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

if __name__ == '__main__':
    dataset = 'cifar10'
    dataset = 'mnist'
    batchSize = 128
    imageSize = 28
    nz=2
    nepoch=1
    if not os.path.exists('./img_VAE-GAN'):
        os.mkdir('./img_VAE-GAN')
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
    vae.load_state_dict(torch.load('./VAE-GAN-VAE_epoch5.pth'))
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
    plt.title('VAE-GAN-MNIST')
    plt.legend()
    plt.show()
    import numpy as np
    def plot_numbers(rangex, rangey):
        numbers = []
        x = torch.arange(rangex[0], rangex[1],(rangex[1]-rangex[0])/10)
        y = torch.arange(rangey[0], rangey[1], (rangey[1]-rangey[0])/10)
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
    plot_numbers([-2, 2], [-2, 2])
