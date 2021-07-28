import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import model
import dataset
import loss
import torchvision
from tensorboardX import SummaryWriter
import os
import time

print("dataloading...")
begin = time.time()
trainset = dataset.MyDataset("./dataset/trainset")
trainset = DataLoader(trainset, batch_size=512, shuffle=True)
end = time.time()
print("dataload success! costing total time:", end-begin,"s")

network = model.Car().cuda().train()

epoch = 500
lr = 0.003
momentum = 0.9

if os.path.exists("runs"):
    os.system("rm -r ./runs")
writer = SummaryWriter('runs')

flag = 100000000

optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, network.parameters()), lr=lr, momentum=momentum)
my_loss = loss.MyLoss()
step_size = int(epoch * 0.3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1 )
for epoch_i in range(1, epoch+1):
    total_loss = 0 
    for imgs, labels in trainset:
        imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        out = network(imgs)
        loss = my_loss(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        print("epoch: ", epoch_i, ", train_loss:%.4f"%loss)
    
    scheduler.step()
    writer.add_scalar('loss', total_loss, global_step=epoch_i)

    if(total_loss<flag):
        flag = total_loss
        torch.save(network, "./weight/best.pth")
 
torch.save(network, "./weight/last.pth")
