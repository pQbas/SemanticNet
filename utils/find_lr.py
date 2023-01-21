import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from dataloader import Hand_Dataset, plot_mini_batch
from torch.utils.data import DataLoader, Dataset, random_split
from model import Segm_Model2
import matplotlib.pyplot as plt

from loss import accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def find_lr(model, optimiser, start_val=1e-6, end_val=1, beta=0.99, loader=None):
    n = len(loader) - 1
    factor = (end_val / start_val) ** (1/n)
    lr = start_val
    optimiser.param_groups[0]['lr'] = lr
    avg_loss, loss, acc = 0., 0., 0.
    lowest_loss = 0.
    batch_num = 0
    losses = []
    long_lrs = []
    accuracies = []
    model = model.to(device=device)

    for i, (x,y) in enumerate(loader, start=1):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        
        optimiser.zero_grad()
        scores = model(x)
        #cost = nn.BCELoss()(scores, y)
        cost = nn.BCEWithLogitsLoss()(scores, y)

        loss = beta*loss + (1-beta)*cost.item()
        avg_loss = loss/(1 - beta**i)
        acc_ = (scores == y).sum() / torch.numel(scores)

        if i > 1 and avg_loss > 4 * lowest_loss:
            print(f'from here{i, cost.item()}')
            return long_lrs, losses, accuracies

        if avg_loss < lowest_loss or i == 1:
            lowest_loss = avg_loss
        
        accuracies.append(acc_.item())
        losses.append(avg_loss)
        long_lrs.append(lr)

        # #step
        cost.backward()
        optimiser.step()

        # #update lr
        print(f'cost: {cost.item():.4f}, lr: {lr:.4f}, acc: {acc_.item():.4f}')
        lr *= factor
        optimiser.param_groups[0]['lr'] = lr

    return long_lrs, losses, accuracies


if __name__ == "__main__":
    PATH = 'C://Users//percy//Documents//FreiHand//dataset'
    TRAIN_PATH = 'C://Users//percy//Documents//FreiHand//dataset//training//rgb3'
    TRAIN_MASKS_PATH = 'C://Users//percy//Documents//FreiHand//dataset//training//mask3'

    transform_image = T.Compose([
        T.Resize([224,224]),
        T.ToTensor()
    ])

    # transform_mask = T.Compose([
    #     T.Resize([224,224]),
    #     T.ToTensor()
    # ])
    
    # Data Loader
    full_dataset = Hand_Dataset(TRAIN_PATH, 
                                TRAIN_MASKS_PATH, 
                                img_transforms = transform_image, 
                                mask_transforms = None)

    BATCH_SIZE = 1
    TRAIN_SIZE = int(len(full_dataset)*0.9)
    VAL_SIZE = len(full_dataset) - TRAIN_SIZE
    print("#Total batches during training --> TRAIN:",TRAIN_SIZE,"| VAL:",VAL_SIZE)

    train_dataset, val_dataset = random_split(full_dataset, [TRAIN_SIZE, VAL_SIZE])
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    imgs, masks = next(iter(train_loader))
    
    plot_mini_batch(imgs, masks, BATCH_SIZE=BATCH_SIZE)

    print("------------------------------------------------------------------------------------------------")

    model = Segm_Model2()

    optimiser_model = torch.optim.SGD(model.parameters(), 
                                    lr=0.01, 
                                    momentum=0.95, 
                                    weight_decay=1e-4)

    lg_lr, losses, accuracies = find_lr(model=model, optimiser = optimiser_model, start_val=1e-6, end_val=1, loader=train_loader)

