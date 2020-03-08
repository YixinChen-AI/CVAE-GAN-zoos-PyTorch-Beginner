import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 784),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间
        )

    def forward(self, x):
        x = self.gen(x)
        return x


G = generator()
if torch.cuda.is_available():
    G = G.cuda()
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
G.load_state_dict(torch.load('./generator.pth'))
z_dimension = 100
z = torch.randn((1, z_dimension)).cuda()
fake_img = G(z)
fake_img = fake_img.view(28,28).detach().cpu()
# fake_img = transforms.ToPILImage(fake_img)
plt.imshow(fake_img,cmap='gray')
plt.axis('off')
plt.show()

