import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL  import Image
import xml.etree.ElementTree as ET

class MyDataset(Dataset):

    def __init__(self, path):
        super().__init__()

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        path = os.path.join(path,"pics")
        self.set_imgPaths(path)
        self.set_imgs()
        self.set_labels()

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        return img, label
    
    def __len__(self):
        return len(self.labels)

    def set_imgPaths(self, path):
        self.img_paths = os.listdir(path)
        self.img_paths.sort()
        for i in range(len(self.img_paths)):
            self.img_paths[i] = os.path.join(path, self.img_paths[i])

    def set_imgs(self):
        self.imgs = []
        for img_path in self.img_paths:
            img = Image.open(img_path)
            img = self.transforms(img)
            self.imgs.append(img)
    
    def set_labels(self):
        self.labels = []
        for img_path in self.img_paths:
            label_path = img_path.replace(".jpg",".xml")
            label_path = label_path.replace("pics","labels")
            if(os.path.exists(label_path)):
                label = self.get_label(label_path)
                label = torch.tensor(label)
                self.labels.append(label)        
            else:
                label = torch.tensor([0., 1.,1.,1.,1.])
                self.labels.append(label)        

    def get_label(self, xml_name):
        tree = ET.ElementTree(file=xml_name)
        root = tree.getroot()
        pic_size = root.find("size")
        pic_width = float(pic_size.find("width").text)
        pic_height = float(pic_size.find("height").text)

        for obj in root.iter("object"):
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        x_center = (xmin+xmax) / 2 / pic_width
        y_center = (ymin+ymax) / 2 / pic_height
        width = (xmax-xmin) / pic_width
        height = (ymax-ymin) / pic_height

        return [1., x_center,y_center,width,height]
