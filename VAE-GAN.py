#######################################################################################################################################
# Ref:
#   Autoencoding beyond pixels using a learned similarity metric, https://arxiv.org/pdf/1512.09300.pdf
#   Auto-Encoding Variational Bayes, https://arxiv.org/abs/1312.6114
#   Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks https://arxiv.org/pdf/1511.06434.pdf
#######################################################################################################################################
from __future__ import print_function
import argparse
import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch.nn.functional as F
import os
# custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#     print(classname)

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)

def loss_function(recon_x,x,mean,logstd):
    # BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
    MSE = MSECriterion(recon_x,x)
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var = torch.pow(torch.exp(logstd),2)
    KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-var)
    return MSE+KLD

if __name__ == '__main__':
    dataset = 'cifar10'
    dataset = 'mnist'
    batchSize = 128
    imageSize = 28
    nz=100
    nepoch=20
    if not os.path.exists('./img_VAE-GAN'):
        os.mkdir('./img_VAE-GAN')
    print("Random Seed: 88")
    random.seed(88)
    torch.manual_seed(88)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 可以优化运行效率
    cudnn.benchmark = True
    if dataset == 'cifar10':
        dataset = dset.CIFAR10(root='./data', download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
                               )
        n_channel = 3
    elif dataset=='mnist':
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
    # vae.load_state_dict(torch.load('./VAE-GAN-VAE.pth'))
    print("=====> 构建D")
    D = Discriminator().to(device)
    # D.load_state_dict(torch.load('./VAE-GAN-Discriminator.pth'))
    criterion = nn.BCELoss().to(device)
    MSECriterion = nn.MSELoss().to(device)

    print("=====> Setup optimizer")
    optimizerD = optim.Adam(D.parameters(), lr=0.0001)
    optimizerVAE = optim.Adam(vae.parameters(), lr=0.0001)

    gen_win = None
    rec_win = None
    print("=====> Begin training")
    for epoch in range(nepoch):
        for i, (data,label) in enumerate(dataloader, 0):
            ###################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###################################################################
            # train with real
            D.zero_grad()
            data = data.to(device)
            label = label.to(device)
            batch_size = data.shape[0]
            output = D(data)
            # print("output size of netDs: ", output.size())
            real_label = torch.ones(batch_size).to(device)  # 定义真实的图片label为1
            fake_label = torch.zeros(batch_size).to(device)  # 定义假的图片的label为0
            errD_real = criterion(output, real_label)
            errD_real.backward()
            real_data_score = output.mean().item()
            # train with fake, taking the noise vector z as the input of D network
            # 随机产生一个潜在变量，然后通过decoder 产生生成图片
            z = torch.randn(batch_size, nz).to(device)
            # 通过vae的decoder把潜在变量z变成虚假图片
            fake_data = vae.decoder_fc(z).view(z.shape[0], 32, 7, 7)
            fake_data = vae.decoder(fake_data)
            output = D(fake_data)
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            # fake_data_score用来输出查看的，是虚假照片的评分，0最假，1为真
            fake_data_score = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()
            ###################################################
            # (2) Update G network which is the decoder of VAE
            ###################################################
            recon_data,mean,logstd = vae(data)
            vae.zero_grad()
            vae_loss = loss_function(recon_data,data,mean,logstd)
            vae_loss.backward(retain_graph=True)
            optimizerVAE.step()
            ###############################################
            # (3) Update G network: maximize log(D(G(z)))
            ###############################################
            vae.zero_grad()
            real_label = torch.ones(batch_size).to(device)  # 定义真实的图片label为1
            output = D(recon_data)
            errVAE = criterion(output, real_label)
            errVAE.backward()
            D_G_z2 = output.mean().item()
            optimizerVAE.step()
            if i%100==0:
                print('[%d/%d][%d/%d] real_score: %.4f fake_score: %.4f '
                      'Loss_D: %.4f '
                      % (epoch, nepoch, i, len(dataloader),
                         real_data_score,
                         fake_data_score,
                         errD.item()))
            if epoch==0:
                real_images = make_grid(data.cpu(), nrow=8, normalize=True).detach()
                save_image(real_images, './img_VAE-GAN/real_images.png')
            sample = torch.randn(80, nz).to(device)
            output = vae.decoder_fc(sample)
            output = vae.decoder(output.view(output.shape[0], 32, 7, 7))
            fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
            save_image(fake_images, './img_VAE-GAN/fake_images-{}.png'.format(epoch + 1))
torch.save(vae.state_dict(), './VAE-GAN-VAE_epoch5.pth')
torch.save(D.state_dict(),'./VAE-GAN-Discriminator_epoch5.pth')


