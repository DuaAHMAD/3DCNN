from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import svd
from numpy.random import normal
        
    
class BDI_3D_Conv(nn.Module):
    def __init__(self, num_classes=2):
        super(BDI_3D_Conv, self).__init__()
        self.conv1_1 = nn.Conv3d(3, 4, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2))
        self.conv1_2 = nn.Conv3d(4, 8, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2))

        self.conv2_1 = nn.Conv3d(8,  8, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2))
        self.conv2_2 = nn.Conv3d(8, 16, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2))

        self.conv3_1 = nn.Conv3d(16, 16, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2))
        self.conv3_2 = nn.Conv3d(16, 32, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2))

        self.conv4_1 = nn.Conv3d(32, 32, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2))
        self.conv4_2 = nn.Conv3d(32, 64, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2))

        self.conv5_1 = nn.Conv3d(64, 64, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2))
        self.conv5_2 = nn.Conv3d(64, 128, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 1))

        self.BN1_1 = nn.BatchNorm3d(4)
        self.BN1_2 = nn.BatchNorm3d(8)

        self.BN2_1 = nn.BatchNorm3d(8)
        self.BN2_2 = nn.BatchNorm3d(16)

        self.BN3_1 = nn.BatchNorm3d(16)
        self.BN3_2 = nn.BatchNorm3d(32)

        self.BN4_1 = nn.BatchNorm3d(32)
        self.BN4_2 = nn.BatchNorm3d(64)

        self.BN5_1 = nn.BatchNorm3d(64)
        self.BN5_2 = nn.BatchNorm3d(128)
        
        self.fc_1 = nn.Linear(128*50*3*2, 100)
        self.fc_2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.BN1_1(F.relu(self.conv1_1(x)))
        x = self.BN1_2(F.relu(self.conv1_2(x)))

        x = self.BN2_1(F.relu(self.conv2_1(x)))
        x = self.BN2_2(F.relu(self.conv2_2(x)))

        x = self.BN3_1(F.relu(self.conv3_1(x)))
        x = self.BN3_2(F.relu(self.conv3_2(x)))

        x = self.BN4_1(F.relu(self.conv4_1(x)))
        x = self.BN4_2(F.relu(self.conv4_2(x)))

        x = self.BN5_1(F.relu(self.conv5_1(x)))
        x = self.BN5_2(F.relu(self.conv5_2(x)))
        
        x = x.reshape(x.size(0), -1)
        
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x
    

class BDI_3D_Conv_Simple(nn.Module):
    def __init__(self, num_classes=2):
        super(BDI_3D_Conv_Simple, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
                       
        self.fc = nn.Linear(3*5*5*64, num_classes)

    def forward(self, out):
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)  
        return out


