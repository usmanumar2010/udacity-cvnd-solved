## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
         # input 1x224x224
        self.conv1 = nn.Conv2d(1, 32, 4)    #32x221x221
        self.pool1 = nn.MaxPool2d(4, 4)     #32x55x55
       
        self.conv2 = nn.Conv2d(32, 64, 3)   #64x53x53
        self.pool2 = nn.MaxPool2d(2, 2)     #64x26x26

        self.conv3 = nn.Conv2d(64, 128, 2)  #128x25x25
        self.pool3 = nn.MaxPool2d(2, 2)     #128x12x12

        self.conv4 = nn.Conv2d(128, 256, 1) #256x12x12
        self.pool4 = nn.MaxPool2d(2, 2)     #256x6x6

        self.lin1 = nn.Linear(256*6*6,1000)
        self.lin2 = nn.Linear(1000,1000)
        self.lin3 = nn.Linear(1000,68*2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        drop1 = nn.Dropout(0.1)
        drop2 = nn.Dropout(0.2)
        drop3 = nn.Dropout(0.3)
        drop4 = nn.Dropout(0.4)
        drop5 = nn.Dropout(0.5)
        drop6 = nn.Dropout(0.6)
        
        x = drop1(self.pool1(F.relu(self.conv1(x))))
        x = drop2(self.pool2(F.relu(self.conv2(x))))
        x = drop3(self.pool3(F.relu(self.conv3(x))))
        x = drop4(self.pool4(F.relu(self.conv4(x))))
        
        x = x.view(x.size(0), -1) # flatten
        
        x = drop5(F.relu(self.lin1(x)))
        x = drop6(self.lin2(x))
        x = self.lin3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
    # AlexNet
class AlexNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        ## Conv layers
        # input of size (1 x 227 x 227)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(4, 4), stride=4, padding=0) # VALID
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2) # SAME
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1) # SAME
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1) # SAME
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1) # SAME
        
        ## Max-Pool layer 
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        ## Linear layers
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=136)
        
        ## Dropout 
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout6 = nn.Dropout(p=0.6)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(num_features=96, eps=1e-05)
        self.bn2 = nn.BatchNorm2d(num_features=256, eps=1e-05)
        self.bn3 = nn.BatchNorm2d(num_features=384, eps=1e-05)
        self.bn4 = nn.BatchNorm2d(num_features=384, eps=1e-05)
        self.bn5 = nn.BatchNorm2d(num_features=256, eps=1e-05)
        self.bn6 = nn.BatchNorm1d(num_features=4096, eps=1e-05)
        self.bn7 = nn.BatchNorm1d(num_features=4096, eps=1e-05)
        
        ## Local response normalization
        # if size=r=2 and a neuron has a strong activation, it will inhibit the activation
        # of the neurons located in the feature maps immediately above and below its own.
#         self.lrn = LocalResponseNorm(size=2, alpha=0.00002, beta=0.75, k=1)  # lrn is on new pytorch version apparently
        
        # Custom weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.xavier_uniform(m.weight, gain=1)
            elif isinstance(m, nn.Linear):
                # FC layers have weights initialized with Glorot uniform initialization
                m.weight = nn.init.xavier_uniform(m.weight, gain=1)

    def forward(self, x):
        
        ## Conv layers
        x = F.elu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = F.elu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
#         x = self.dropout2(x)
        
        x = F.elu(self.conv3(x))
        x = self.bn3(x)
        x = self.dropout4(x)
        
        x = F.elu(self.conv4(x))
        x = self.bn4(x)
        x = self.dropout4(x)
        
        x = F.elu(self.conv5(x))
        x = self.bn5(x)
        x = self.pool(x)
#         x = self.dropout4(x)

        ## Flatten
        x = x.view(x.size(0), -1) 
        
        ## Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.bn6(x)
        x = self.dropout6(x)
        
        x = F.elu(self.fc2(x))
        x = self.bn6(x)
        x = self.dropout6(x)
        
#         x = F.tanh(self.fc3(x))
        x = self.fc3(x)
    
        return x
    
    #vgg
class vgg(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # stride = 1, by default and Kernel size is
                                                                            #3 x 3 a 3 by 3 filter will revolve
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # the formula used over here is ((W-F)+2P)/S +1 weher W is width of our image and is F Filter=kernel=3, so ((64-3)+2(1))/1 +1=64 output
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1) #((64-3)+2(1))/1 +1
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

        # linear layer (512 * 7 * 7 ->4096)  or 25088 -> 4096
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)

        self.fc3 = nn.Linear(in_features=4096, out_features=136) 

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.5)

        ## Dropout 
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.4)
        self.dropout3 = nn.Dropout(p=0.6)

        # Batch Normalization

        self.bn1 = nn.BatchNorm1d(num_features=4096, eps=1e-05)
        self.bn2 = nn.BatchNorm1d(num_features=4096, eps=1e-05)


          ## Local response normalization
    # if size=r=2 and a neuron has a strong activation, it will inhibit the activation
    # of the neurons located in the feature maps immediately above and below its own.
#         self.lrn = LocalResponseNorm(size=2, alpha=0.00002, beta=0.75, k=1)  # lrn is on new pytorch version apparently

        # Custom weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.xavier_uniform(m.weight, gain=1)
            elif isinstance(m, nn.Linear):
                # FC layers have weights initialized with Glorot uniform initialization
                m.weight = nn.init.xavier_uniform(m.weight, gain=1)

        
    def forward(self, x):

        out = F.relu(self.conv1_1(x))  
        out = F.relu(self.conv1_2(out))  
        out = self.pool1(out)  
        out = self.dropout1(out)

        out = F.relu(self.conv2_1(out))  
        out = F.relu(self.conv2_2(out))  
        out = self.pool2(out)  

        out = F.relu(self.conv3_1(out))  
        out = F.relu(self.conv3_2(out))  
        out = F.relu(self.conv3_3(out))  
        out = self.pool3(out)  
        out = self.dropout2(out)

        out = F.relu(self.conv4_1(out))  
        out = F.relu(self.conv4_2(out))  
        out = F.relu(self.conv4_3(out))  
        out = self.pool4(out)  

        out = F.relu(self.conv5_1(out))  
        out = F.relu(self.conv5_2(out))  
        out = F.relu(self.conv5_3(out))  
        out = self.pool5(out)
        out = self.dropout3(out)

             ## Flatten
        out = out.view(out.size(0), -1) 

        ## Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.bn1(out)
        out = self.dropout3(out)

        out = F.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.dropout3(out)

#         x = F.tanh(self.fc3(x))
        out = self.fc3(out)



        return out

        
        
        
      