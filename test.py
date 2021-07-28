import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import model
import dataset
import torchvision
import os
import time

print("dataloading...")
begin = time.time()
trainset = dataset.MyDataset("./dataset/trainset")
trainset = DataLoader(trainset, batch_size=50, shuffle=True)
end = time.time()
print("dataload success! costing total time:", end-begin,"s")

network = model.Car().cuda().eval()

for imgs, labels in trainset:
    imgs, labels = imgs.cuda(), labels.cuda()
    out = network(imgs)
