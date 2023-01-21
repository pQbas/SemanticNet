import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from models.FullyConvNet import Segm_Model2
from models.SegNet import SegNet

from dataloaders.dataloader import Hand_Dataset, plot_mini_batch

from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

from utils.loss import accuracy


class Trainer:
    def __init__(self):
        
        self.dataset = None
        self.dataset_size = None
        
        self.train_set = None
        self.train_size = None
        self.train_loader = None
        
        self.val_set = None
        self.val_size = None
        self.val_loader = None
        
        self.optimiser = None
        self.model = None
        self.scheduler = None
        
        return
    
    def plot(self, batch_size):
        imgs, masks = next(iter(self.train_loader)) 
        plot_mini_batch(imgs, masks, batch_size)    
        return
    
    def set_dataset(self, dataset):
        self.dataset = dataset
        return
    
    def split_dataset(self, train_percentage, batch_size):
        self.train_size = int(len(self.dataset)*train_percentage)
        self.val_size = len(self.dataset) - self.train_size
        self.train_set, self.val_set = random_split(self.dataset, [self.train_size, self.val_size])
        
        self.train_loader = DataLoader(self.train_set, batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size, shuffle=True)
        
        return
    
    def set_optimiser(self, optimiser):
        self.optimiser = optimiser
        return
    
    def set_model(self, model):
        self.model = model
        return
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        return

    def train(self, epochs = 100, store_every = 1, save_path=None, device=None):

        model = self.model.to(device=device)
        scheduler = self.scheduler 
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            
            train_correct_num = 0
            train_total = 0
            train_cost_acum = 0.0
            model.train()

            for mb, (x,y) in enumerate(self.train_loader, start=1):
                
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                scores = model(x)
                scores = torch.Tensor.type(scores, dtype=torch.float32)

                cost = criterion(scores[:,1,:,:], y[:,1,:,:])
                self.optimiser.zero_grad()
                cost.backward()
                self.optimiser.step()

                if scheduler: 
                    scheduler.step()
                
                train_predictions = torch.argmax(scores, dim=1)

                train_correct_num += (train_predictions == y[:,1,:,:]).sum()
                train_total += torch.numel(train_predictions)
                train_cost_acum += cost.item()

                if mb%store_every == 0:

                    val_cost, val_acc, dice, iou = accuracy(self.model, self.val_loader)
                    train_acc = float(train_correct_num)/train_total
                    train_cost_every = float(train_cost_acum)/mb                
                    print(f'epoch: {epoch}, mb: {mb}/{len(self.train_loader)}, train cost: {train_cost_every:.4f}, val cost: {val_cost:.4f}, train acc: {train_acc:.10f},  val acc: {val_acc:.10f}, dice: {dice:.10f}, iou: {iou:.10f}')
        
        torch.save(model.state_dict(), save_path)
                
        return self.model


if __name__ == "__main__":
    torch.manual_seed(42)

    BATCH_SIZE = 32
    TRAIN_PERCENTAGE = 0.8
    EPOCHS = 50
    LEARNING_RATE = 5*1e-3

    
    TRAIN_PATH = './dataset/training/rgb3'
    TRAIN_MASKS_PATH = './dataset/training/mask3'
    NAME = f'SegNet_{EPOCHS}epochs_{BATCH_SIZE}batch.pth'
    SAVE_PATH = f'./weights/{NAME}'
        
    #----------------------------------------------------------------------
    dataset = Hand_Dataset(TRAIN_PATH, 
                           TRAIN_MASKS_PATH, 
                           img_transforms = T.Compose([T.Resize([32,32]), T.ToTensor()]), 
                           mask_transforms = T.Compose([T.Resize([32,32])]))
    
    
    model = SegNet()  # Segm_Model2()
    
    
    optimiser = torch.optim.SGD(model.parameters(), 
                                lr=LEARNING_RATE, 
                                momentum=0.5,
                                weight_decay=LEARNING_RATE*1e-2)
    
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser,
                                                    max_lr = LEARNING_RATE,
                                                    steps_per_epoch = int(len(dataset)*TRAIN_PERCENTAGE),
                                                    epochs = EPOCHS,
                                                    pct_start=0.43,
                                                    div_factor=10,
                                                    final_div_factor=1000,
                                                    three_phase=True)
    
    #----------------------------------------------------------------------
    
    trainer = Trainer()
    trainer.set_dataset(dataset)
    trainer.split_dataset(TRAIN_PERCENTAGE, BATCH_SIZE)
    trainer.plot(BATCH_SIZE)
    trainer.set_model(model)
    trainer.set_optimiser(optimiser)
    trainer.set_scheduler(scheduler)
    trainer.train(save_path= SAVE_PATH, 
                  device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    