import torch
import torchvision

trainset = torchvision.datasets.FashionMNIST(root, train=True, transform=None, 
											target_transform=None, download=True)

testset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, 
											target_transform=None, download=True)
