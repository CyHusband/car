import torch
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

transforms_ = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.toTensor(),
    transforms.Normalize((0.485,0.456,0.486), (0.229,0.224,0.225)),
]) 

def detect(pic_name, detect_model):
    
    pic = Image.open(pic_name)
    pic = transforms_(pici.permute(2,0,1)).unsqueeze(dim=0).cuda()

    out = detect_model(pic)
    return out
# get all labels
def get_labels():
    test_label_path = "testset/labels"
    label_paths = os.listdir(test_label_path)
    
    labels = []
    for label_path in label_paths:
        label_path = os.path.join(test_label_path,label_path)
        temp = []
        with open(label_path,"r") as f:
            for cell in f.read()[2:-1]:
                temp.append(float(cell))
        labels.append(temp)

    return labels 

def main():
    detect_model = torch.load("./weight/best.pth").cuda.eval() 
    labels = get_labels()

    test_pic_path = "testset/images"
    pic_names = os.listdir(test_path)
    pic_names.sort()

    total = len(pic_names)
    n = 1 
    total_loss = 0

    for i in range(total):
        pic_name = os.path.join(test_pic_path, pic_names[i])
        out = detect(pic_name, detect_model)
        loss = F.mseloss(out, labels[i])
        total_loss += loss

        print(n, "/", total, ",", labels[i]," detection result: ", out, ", loss:", loss)
        n +=1

    print("total_loss:", total_loss, " ,mean_loss", total_loss/total)

main()
