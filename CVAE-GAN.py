import random
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
import os
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder_conv = nn.Sequential(
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
        self.decoder_fc = nn.Linear(nz+10,32 * 7 * 7)
        self.decoder_deconv = nn.Sequential(
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
        z = self.encoder(x)
        output = self.decoder(z)
        return output
    def encoder(self,x):
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        return z,mean,logstd
    def decoder(self,z):
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder_deconv(out3)
        return out3
class Discriminator(nn.Module):
    def __init__(self,outputn=1):
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
            nn.Linear(1024, outputn),
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
    if not os.path.exists('./img_CVAE-GAN'):
        os.mkdir('./img_CVAE-GAN')
    print("Random Seed: 88")
    random.seed(88)
    torch.manual_seed(88)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    vae.load_state_dict(torch.load('./CVAE-GAN-VAE.pth'))
    print("=====> 构建D")
    D = Discriminator(1).to(device)
    D.load_state_dict(torch.load('./CVAE-GAN-Discriminator.pth'))
    print("=====> 构建C")
    C = Discriminator(10).to(device)
    C.load_state_dict(torch.load('./CVAE-GAN-Classifier.pth'))
    criterion = nn.BCELoss().to(device)
    MSECriterion = nn.MSELoss().to(device)

    print("=====> Setup optimizer")
    optimizerD = optim.Adam(D.parameters(), lr=0.0001)
    optimizerC = optim.Adam(C.parameters(), lr=0.0001)
    optimizerVAE = optim.Adam(vae.parameters(), lr=0.0001)

    for epoch in range(nepoch):
        for i, (data,label) in enumerate(dataloader, 0):
            # 先处理一下数据
            data = data.to(device)
            label_onehot = torch.zeros((data.shape[0], 10)).to(device)
            label_onehot[torch.arange(data.shape[0]), label] = 1
            batch_size = data.shape[0]
            # 先训练C
            output = C(data)
            real_label = label_onehot.to(device)  # 定义真实的图片label为1
            errC = criterion(output, real_label)
            C.zero_grad()
            errC.backward()
            optimizerC.step()
            # 再训练D
            output = D(data)
            real_label = torch.ones(batch_size).to(device)   # 定义真实的图片label为1
            fake_label = torch.zeros(batch_size).to(device)  # 定义假的图片的label为0
            errD_real = criterion(output, real_label)

            z = torch.randn(batch_size, nz + 10).to(device)
            fake_data = vae.decoder(z)
            output = D(fake_data)
            errD_fake = criterion(output, fake_label)

            errD = errD_real+errD_fake
            D.zero_grad()
            errD.backward()
            optimizerD.step()
            # 更新VAE(G)1
            z,mean,logstd = vae.encoder(data)
            z = torch.cat([z,label_onehot],1)
            recon_data = vae.decoder(z)
            vae_loss1 = loss_function(recon_data,data,mean,logstd)
            # 更新VAE(G)2
            output = D(recon_data)
            real_label = torch.ones(batch_size).to(device)
            vae_loss2 = criterion(output,real_label)
            # 更新VAE(G)3
            output = C(recon_data)
            real_label = label_onehot
            vae_loss3 = criterion(output, real_label)

            vae.zero_grad()
            vae_loss = vae_loss1+vae_loss2+vae_loss3
            vae_loss.backward()
            optimizerVAE.step()
            if i%100==0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_C: %.4f Loss_G: %.4f'
                      % (epoch, nepoch, i, len(dataloader),
                         errD.item(),errC.item(),vae_loss.item()))
            if epoch==0:
                real_images = make_grid(data.cpu(), nrow=8, normalize=True).detach()
                save_image(real_images, './img_CVAE-GAN/real_images.png')
            if i == len(dataloader)-1:
                sample = torch.randn(data.shape[0], nz).to(device)
                print(label)
                sample = torch.cat([sample,real_label],1)
                output = vae.decoder(sample)
                fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
                save_image(fake_images, './img_CVAE-GAN/fake_images-{}.png'.format(epoch + 26))
torch.save(vae.state_dict(), './CVAE-GAN-VAE.pth')
torch.save(D.state_dict(),'./CVAE-GAN-Discriminator.pth')
torch.save(C.state_dict(),'./CVAE-GAN-Classifier.pth')



