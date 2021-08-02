import torch
from torch.utils.data import DataLoader, Dataset
import model
import dataset_test
import os
import time

def get_pre_count(out, labels):
    cls_pre = out[:, 0]
    cls_tar = labels[:, 0]
    cls = cls_tar > 0.5

    box_pre = out[:, 1:]
    box_tar = labels[:, 1:]
    iou = get_iou(box_pre, box_tar)
    
    result = (cls * iou).sum()
    return result

def get_iou(box1, box2, eps=1e-12, thresthold=0.5):
    #xywh
    b1_x1, b1_x2 = box1[:,0] - box1[:,2] / 2, box1[:,0] + box1[:,2] / 2 + eps 
    b1_y1, b1_y2 = box1[:,1] - box1[:,3] / 2, box1[:,1] + box1[:,3] / 2 + eps 
    b2_x1, b2_x2 = box2[:,0] - box2[:,2] / 2, box2[:,0] + box2[:,2] / 2 + eps 
    b2_y1, b2_y2 = box2[:,1] - box2[:,3] / 2, box2[:,1] + box2[:,3] / 2 + eps 

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
        (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter

    iou = inter / union  # iou
    return iou > thresthold


print("dataloading...")
begin = time.time()
trainset = dataset_test.MyDataset("./dataset/testset")
trainset = DataLoader(trainset, batch_size=50, shuffle=True)
end = time.time()
print("dataload success! costing total time:", end-begin,"s")

network = torch.load("./weight/best.pth").cuda().eval()

count = 0 
for imgs, labels in trainset:
    imgs, labels = imgs.cuda(), labels.cuda()
    out = network(imgs)
    count += get_pre_count(out, labels)
print(count)


