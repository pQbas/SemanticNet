import torch
import torch.nn as nn
from .utils import intersection_over_union

import torch
import torch.nn.functional as F
# from .dataloader import Hand_Dataset, transform_image, transform_mask
from torch.utils.data import DataLoader, Dataset, random_split
# from .model import Segm_Model2
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



def iou(preds, y):
    intersection = (preds*y).sum()
    union = (preds + y - preds*y).sum()
    iou_ = (intersection)/(union + 1e-8)
    return iou_

def dice(preds, y):
    intersection = (preds*y).sum()
    denom = (preds + y).sum()
    dice_ = 2*intersection/(denom + 1e-8)
    return dice_

def accuracy(model, loader):
    correct = 0
    intersection = 0
    denom = 0
    union = 0
    dice_metric = 0.0
    iou_metric = 0.0
    total = 0
    cost = 0

    #model = model.to(device=device)

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32).squeeze(1)
            scores = model(x)
            scores = scores.squeeze(1)

            #cost += (nn.BCEWithLogitsLoss()(scores, y)).item()
            cost += (F.cross_entropy(input=scores, target=y)).item()
            
            # standar accuracy is not optimal
            preds = torch.argmax(scores, dim=1)
            
            #accuracy_matrix = torch.Tensor.numpy((preds).cpu())[0,:,:] #== y[:,1,:,:]
            #plt.imshow(accuracy_matrix)
            #plt.show()

            correct += (preds == y[:,1,:,:]).sum()
            total += torch.numel(preds)   
            iou_metric += iou(preds, y[:,1,:,:])
            dice_metric += dice(preds, y[:,1,:,:])  

    return cost/len(loader), float(correct)/total, dice_metric, iou_metric


if __name__=="__main__":

    scores = torch.tensor( [[[[1.0, 1.0], 
                              [1.0, 0.0]]]], dtype=torch.float32)

    y = torch.tensor(      [[[[1.0, 1.0], 
                              [1.0, 1.0]]]], dtype=torch.float32)

    print(scores)
    print(y)
    #cost = nn.BCELoss()(scores, y).item()
    #cost = F.cross_entropy(scores, y)
    #print(cost)
    print('dice:',dice(scores, y))
