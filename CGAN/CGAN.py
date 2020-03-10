# coding=utf-8
import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os

# 创建文件夹
if not os.path.exists('./img_CGAN'):
    os.mkdir('./img_CGAN')
# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
num_epoch = 25
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
# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.dis(x)
        return x
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
D = discriminator()
G = generator()
D = D.to(device)
G = G.to(device)
# 载入模型
# G.load_state_dict(torch.load('./generator_CGAN_z100.pth'))
# D.load_state_dict(torch.load('./discriminator_CGAN_z100.pth'))
#########判别器训练train#####################
criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
###########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (img, label) in enumerate(dataloader):
        num_img = img.size(0)
        label_onehot = torch.zeros((num_img,10)).to(device)
        label_onehot[torch.arange(num_img),label]=1
        # view()函数作用把img变成[batch_size,channel_size,784]
        img = img.view(num_img,  -1)  # 将图片展开为28*28=784
        real_img = img.to(device)
        real_label = label_onehot
        fake_label = torch.zeros((num_img,10)).to(device)
        # print(img.shape)
        # 计算真实图片的损失
        real_out = D(real_img)  # 将真实图片放入判别器中
        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
        # 计算假的图片的损失
        z = torch.randn(num_img, z_dimension+10).to(device)  # 随机生成一些噪声
        fake_img = G(z)  # 随机噪声放入生成网络中，生成一张假的图片
        fake_out = D(fake_img)  # 判别器判断假的图片
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss

        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数

        # ==================训练生成器============================
        ################################生成网络的训练###############################
        z = torch.randn(num_img, z_dimension).to(device) # 得到随机噪声
        z = torch.cat([z, real_label],1)
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)
        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
        # 打印中间的损失
        # try:
        if (i + 1) % 100 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '.format(
                epoch, num_epoch, d_loss.item(), g_loss.item(),
            ))
        # except BaseException as e:
        #     pass
        if epoch == 0:
            real_images = real_img.cpu().clamp(0,1).view(-1,1,28,28).data
            save_image(real_images, './img_CGAN/real_images.png')
        if i == len(dataloader)-1:
            fake_images = fake_img.cpu().clamp(0,1).view(-1,1,28,28).data
            save_image(fake_images, './img_CGAN/fake_images-{}.png'.format(epoch + 1))
# 保存模型
torch.save(G.state_dict(), './generator_CGAN_z100.pth')
torch.save(D.state_dict(), './discriminator_CGAN_z100.pth')