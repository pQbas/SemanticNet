import torch
import torch.nn as nn

from models.FullyConvNet import Segm_Model2
from models.SegNet import SegNet

# from train import find_lr, accuracy, train
import matplotlib.pyplot as plt
from dataloaders.dataloader import plot_mini_batch, Hand_Dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torchvision import transforms as T
from PIL import Image

transform_image = T.Compose([
    T.Resize([32,32]),
    T.ToTensor(),
])

transform_inverse = T.Compose([
    T.Resize([224,224]),
])

model = SegNet() #Segm_Model2()

image_name = './dataset/training/rgb3/00000003.jpg'
img = Image.open(image_name)
trans = T.ToTensor()

SAVE_PATH = './weights/SegNet_50epochs_32batch.pth'

with torch.no_grad():
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

    input = transform_image(img).reshape((1,3,32,32))
    
    output = model(input)
    print(output.shape)
    

    mask = torch.Tensor.numpy(output[0,1,:,:])
    mask = cv2.resize(mask, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    plt.imshow(img)
    plt.axis('Off')
    plt.imshow(mask, alpha=0.5)

plt.show()