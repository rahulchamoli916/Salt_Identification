import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

class convBlock(nn.Module):
  def __init__(self, in_channels, filters, size, stride = 1, activation = True):
    super(convBlock, self).__init__()
    # initializing variables
    self.activation = activation
    # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.conv = nn.Conv2d(in_channels, filters, size, stride = stride, padding = size//2)
    # nn.BatchNorm2d(num_features)
    self.norm = nn.BatchNorm2d(filters)

  def forward(self, x):
    x = self.conv(x)
    x = self.norm(x)
    if self.activation:
      # nn.functional.relu(input)
      return F.relu(x)
    else:
      return x

class residualBlock(nn.Module):
  def __init__(self, in_channels, filters, size = 3):
    super(residualBlock, self).__init__()
    # initializing variables
    # # nn.BatchNorm2d(num_features)
    self.norm = nn.BatchNorm2d(in_channels)
    self.conv1 = convBlock(in_channels, filters, size)
    self.conv2 = convBlock(filters, filters, size, activation=False)

  def forward(self, x):
    residual = x  
    # nn.functional.relu(input)
    x = F.relu(x)
    x = self.norm(x)
    x = self.conv1(x)
    x = self.conv2(x)
    return x 

class deconvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size = 2, stride = 2):
    super(deconvBlock, self).__init__()
    # initializing variables
    # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
    self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride)

  def forward(self, x1, x2):
    xd = self.deconv(x1)
    # cat is used to concatenate sequence of tensors, dim=0 means row-wise and dim=1 means column-wise
    x = torch.cat([xd, x2], dim = 1)
    return x


class UnetModel(nn.Module):
    def __init__(self, filters = 16, dropout = 0.5):
        super(UnetModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, filters, 3, padding = 1),
            residualBlock(filters, filters),
            residualBlock(filters, filters),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout/2),
            nn.Conv2d(filters, filters * 2, 3, padding = 1),
            residualBlock(filters * 2, filters * 2),
            residualBlock(filters * 2, filters * 2),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            nn.Conv2d(filters * 2, filters * 4, 3, padding = 1),
            residualBlock(filters * 4, filters * 4),
            residualBlock(filters * 4, filters * 4),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            nn.Conv2d(filters * 4, filters * 8, 3, padding = 1),
            residualBlock(filters * 8, filters * 8),
            residualBlock(filters * 8, filters * 8),
            nn.ReLU()
        )
            

        self.middle = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            nn.Conv2d(filters * 8, filters * 16, 3, padding = 3//2),
            residualBlock(filters * 16, filters * 16),
            residualBlock(filters * 16, filters * 16),
            nn.ReLU()
        )
        
        self.deconv4 = deconvBlock(filters * 16, filters * 8, 2)
        self.upconv4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 16, filters * 8, 3, padding = 1),
            residualBlock(filters * 8, filters * 8),
            residualBlock(filters * 8, filters * 8),
            nn.ReLU()
        )
  

        self.deconv3 = deconvBlock(filters * 8, filters * 4, 3)
        self.upconv3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 8, filters * 4, 3, padding = 1),
            residualBlock(filters * 4, filters * 4),
            residualBlock(filters * 4, filters * 4),
            nn.ReLU()
        )
        
        self.deconv2 = deconvBlock(filters * 4, filters * 2, 2)
        self.upconv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 4, filters * 2, 3, padding = 1),
            residualBlock(filters * 2, filters * 2),
            residualBlock(filters * 2, filters * 2),
            nn.ReLU()
        )

        self.deconv1 = deconvBlock(filters * 2, filters, 3)
        self.upconv1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 2, filters, 3, padding = 1),
            residualBlock(filters, filters),
            residualBlock(filters, filters),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Conv2d(filters, 1, 3, padding = 1)
        )
        
    def forward(self, x):
        conv1 = self.conv1(x) 
        # 101 -> 50
        conv2 = self.conv2(conv1) 
        # 50 -> 25
        conv3 = self.conv3(conv2) 
        # 25 -> 12
        conv4 = self.conv4(conv3) 
        # 12 - 6
        x = self.middle(conv4) 
        
        # 6 -> 12
        x = self.deconv4(x, conv4)
        x = self.upconv4(x)
        # 12 -> 25
        x = self.deconv3(x, conv3)
        x = self.upconv3(x)
        # 25 -> 50
        x = self.deconv2(x, conv2)
        x = self.upconv2(x)
        # 50 -> 101
        x = self.deconv1(x, conv1)
        x = self.upconv1(x)

        return x