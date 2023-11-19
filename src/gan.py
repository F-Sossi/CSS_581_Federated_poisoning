import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Directory for saving models
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)


# Generator Definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Discriminator Definition
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Final layer to reduce to a single scalar value
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),  
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        # Flatten the output to shape [batch_size]
        print(output.size())  # To check the size before flattening
        return output.view(-1)




# Create instances of the generator and discriminator
netG = Generator()
netD = Discriminator()

# Configure device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG.to(device)
netD.to(device)

# Set up Optimizers
lr = 0.0002
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Load CIFAR-10 Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Training Loop
num_epochs = 5  # Adjust this for better results

for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):

        ############################
        # (1) Update Discriminator #
        ############################

        # Train with real images
        netD.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        labels = torch.full((batch_size,), 1., dtype=torch.float, device=device)

        output = netD(real_data)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake images
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_data = netG(noise)
        labels.fill_(0.)  # Fake labels are 0
        output = netD(fake_data.detach())
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        ##########################
        # (2) Update Generator   #
        ##########################

        netG.zero_grad()
        labels.fill_(1.)  # We want to fool the discriminator, so we label them as real
        output = netD(fake_data)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print(
                f'[{epoch}/{num_epochs}][{i}/{len(trainloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

    # Save Models at the end of each epoch
    torch.save(netG.state_dict(), os.path.join(model_dir, f'generator_epoch_{epoch}.pth'))
    torch.save(netD.state_dict(), os.path.join(model_dir, f'discriminator_epoch_{epoch}.pth'))

print("Training complete. Models saved.")


