# coding=utf-8
import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
# 创建文件夹
if not os.path.exists('./img_VAE'):
    os.mkdir('./img_VAE')
# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_img(x):
    # out = 0.5 * (x+0.5)
    img = make_grid(x, nrow=8, normalize=True).detach()
    # out = x.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    # out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return img

batch_size = 64
num_epoch = 15
z_dimension = 2
# 图形啊处理过程
img_transform = transforms.Compose([
    transforms.ToTensor(),
])

# mnist dataset mnist数据集下载
mnist = datasets.MNIST(root='./data/', train=True, transform=img_transform, download=True)

# data loader 数据载入
dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

# 定义判别器  #####Discriminator######使用多层网络来作为判别器
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
        self.encoder_fc1=nn.Linear(32*7*7,z_dimension)
        self.encoder_fc2=nn.Linear(32*7*7,z_dimension)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(z_dimension,32 * 7 * 7)
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

def loss_function(recon_x,x,mean,std):
    BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var = torch.pow(torch.exp(std),2)
    KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-var)
    return BCE+KLD
# 创建对象
vae = VAE().to(device)
# vae.load_state_dict(torch.load('./VAE_z2.pth'))
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
###########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # view()函数作用把img变成[batch_size,channel_size,784]
        img = img.view(num_img,  1,28,28).to(device)  # 将图片展开为28*28=784
        x,mean,std = vae(img)  # 将真实图片放入判别器中
        loss = loss_function(x,img,mean,std)
        vae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        loss.backward()  # 将误差反向传播
        vae_optimizer.step()  # 更新参数
        # try:
        if (i + 1) % 100 == 0:
            print('Epoch[{}/{}],vae_loss:{:.6f} '.format(
                epoch, num_epoch, loss.item(),
            ))

        if epoch == 0:
            real_images = make_grid(img.cpu(), nrow=8, normalize=True).detach()
            save_image(real_images, './img_VAE/real_images.png')
        sample = torch.randn(64,z_dimension).to(device)
        output = vae.decoder_fc(sample)
        output = vae.decoder(output.view(output.shape[0],32,7,7))
        fake_images = make_grid(x.cpu(), nrow=8, normalize=True).detach()
        save_image(fake_images, './img_VAE/fake_images-{}.png'.format(epoch + 16))
# 保存模型
torch.save(vae.state_dict(), './VAE_z2.pth')
