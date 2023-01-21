import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.filter_bank = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()    
        )        
        self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  
        
    def forward(self, x):
        x = self.filter_bank(x)
        x, indices = self.pooling(x)
        return x, indices


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.upsampling = nn.MaxUnpool2d(2, stride=2)
        self.filter_bank = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x, pooling_indices):        
        x = self.upsampling(x, indices=pooling_indices)
        x = self.filter_bank(x)
        return x


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()     
        # operations
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        
        # arquitecture
        self.encoder1 = Encoder(3,8)
        self.encoder2 = Encoder(8,16)
        self.encoder3 = Encoder(16,32)
        self.decoder1 = Decoder(32,16)
        self.decoder2 = Decoder(16,8)
        self.decoder3 = Decoder(8,2)
        
        # intializing weights
        self.initialize_weights()
   
        
    def forward(self, x):
        x, pool_indices1 = self.encoder1(x)
        x, pool_indices2 = self.encoder2(x)
        x, pool_indices3 = self.encoder3(x)
        x = self.decoder1(x, pool_indices3)
        x = self.decoder2(x, pool_indices2)
        x = self.decoder3(x, pool_indices1)
        x = self.softmax(x)
        return x
    
    
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                        
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 1)
                    
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    


if __name__=='__main__':
    
    model = SegNet()
    
    with torch.no_grad():
        x = torch.rand((1,3,32,32))
        y = model(x)
        print(y.shape)     
        