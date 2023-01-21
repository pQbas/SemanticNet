import torch
import torch.nn.functional as F


class Segm_Model1(torch.nn.Module):

    def __init__(self):

        super(Segm_Model1, self).__init__()

        # utilities
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.adativepool = torch.nn.AdaptiveAvgPool2d(1)
        

        # 1st - stack of convolutional layers
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        # 2nd - stack of convolutional layers
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 3rd - stack of convolutional layers
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 4td - stack of upsampling layers
        self.conv7 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.initialize_weights()



    def upsampling(self, x):
        
        N, C, H, W = x.shape
        #x = torch.reshape(x, shape=(1, C, H, W))
        x = F.interpolate(x, scale_factor=(2,2), mode='bilinear')

        print(x.shape)
        #N, C, H, W = x.shape
        #x = torch.reshape(x, shape=(C, H, W))
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



    def forward(self, x):

        # ----------------- Feature Detection Stage --------------------
        # 1st - stack of convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.maxpool(x)


        # 2nd - stack of convolutional layers
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)

        x = self.maxpool(x)


        # 3rd - stack of convolutional layers
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)


        # -------------------- Segmentation Stage ------------------------------------

        # 4th - stack of convolutional layers
        x = self.conv7(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=(2,2), mode='bilinear')

        x = self.conv8(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=(2,2), mode='bilinear')
        
        x = self.conv9(x)
        x = self.relu(x)
        
        return x
    


class Segm_Model2(torch.nn.Module):
    def __init__(self):
        super(Segm_Model2, self).__init__()

        # utilities
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.adativepool = torch.nn.AdaptiveAvgPool2d(1)
        
        # 1st - stack of convolutional layers
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0)


        # 3rd - stack of convolutional layers
        self.pred1 = torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.pred2 = torch.nn.Conv2d(in_channels=24, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.pred3 = torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

        # 4th - Deconvolutions
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        
        
        
        self.initialize_weights()


    def forward(self, x):

        # ----------------- Feature Detection Stage --------------------
        # 1st - Backbone
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        pool1 = self.maxpool(x)
        x = self.conv3(pool1)
        x = self.relu(x)
        pool2 = self.maxpool(x)
        x = self.conv4(pool2)
        x = self.relu(x)
        pool3 = self.maxpool(x)

        # 2nd - Prediction
        pool1_pred = self.pred1(pool1)
        pool2_pred = self.pred2(pool2)
        pool3_pred = self.pred3(pool3)

        # 3rd - Combination
        x = self.deconv3(pool3_pred)
        x = torch.add(pool2_pred, x)
        #x = self.relu(x)

        x = self.deconv2(x)
        x = torch.add(pool1_pred, x)
        #x = self.relu(x)
        
        x = self.deconv1(x)
        #
        x = self.conv5(x)
        #x = self.relu(x)
        x = self.softmax(x)
        #x = self.sigmoid(x)
        
        return x

    def upsampling(self, x):
        
        N, C, H, W = x.shape
        #x = torch.reshape(x, shape=(1, C, H, W))
        x = F.interpolate(x, scale_factor=(2,2), mode='bilinear')

        #print(x.shape)
        #N, C, H, W = x.shape
        #x = torch.reshape(x, shape=(C, H, W))
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



if __name__=="__main__":

    my_segmentation_model = Segm_Model2()

    x = torch.rand(1, 3, 8, 8)
    y = my_segmentation_model(x)

    print(y)

    print(x.shape)
    print(y.shape)


