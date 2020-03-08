import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
class generator(nn.Module):
    def __init__(self,input_size,num_feature):
        super(generator, self).__init__()
        self.fc=nn.Linear(input_size,num_feature)
        self.br=nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )
        self.gen = nn.Sequential(
            nn.Conv2d(1,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32,1,3,stride=2,padding=1),
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.fc(x)
        x=x.view(x.shape[0],1,56,56)
        x=self.br(x)
        x=self.gen(x)
        return x
G = generator(100,1*56*56)
if torch.cuda.is_available():
    G = G.cuda()
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
G.load_state_dict(torch.load('./generator_WDCGAN-GP.pth'))
z_dimension = 100
z = torch.randn((1, z_dimension)).cuda()
fake_img = G(z)
fake_img = fake_img.view(28,28).detach().cpu()
fake_img1 = fake_img*0.5+0.5
# fake_img = transforms.ToPILImage(fake_img)
plt.imshow(fake_img,cmap='gray')
plt.axis('off')
plt.show()
plt.imshow(fake_img1,cmap='gray')
plt.axis('off')
plt.show()
print(fake_img[0])
print(fake_img1[0])

