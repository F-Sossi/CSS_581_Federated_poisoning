import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Load the poisoned data
poisoned_data = torch.load('../fakeData/poisoned_data_epoch_004.pt')  # adjust the filename as necessary

# Resize the poisoned data to match the CIFAR-10 data
poisoned_data_resized = F.interpolate(poisoned_data, size=(32, 32))

print('Resized poisoned data shape:', poisoned_data_resized.shape)

# Load a batch of the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
cifar10_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
cifar10_loader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=64, shuffle=True)
cifar10_data_iter = iter(cifar10_loader)
cifar10_data = next(cifar10_data_iter)[0]

# Print out the shapes of the datasets
print('Poisoned data shape:', poisoned_data_resized.shape)
print('CIFAR-10 data shape:', cifar10_data.shape)

# Display some of the poisoned images
fig, axs = plt.subplots(1, 5)
for i in range(5):
    axs[i].imshow(poisoned_data_resized[i].cpu().permute(1, 2, 0))
plt.show()

# Display some of the CIFAR-10 images
fig, axs = plt.subplots(1, 5)
for i in range(5):
    axs[i].imshow(cifar10_data[i].cpu().permute(1, 2, 0))
plt.show()