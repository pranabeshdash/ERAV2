import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model_1_4(nn.Module):

    def __init__(self, dropout_value=0):
        super(Model_1_4, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            #nn.Dropout(dropout_value)
        ) # input_size = 28x28x1, output_size = 26x26x8, RF = 3x3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            #nn.Dropout(dropout_value)
        ) # input_size = 26x26x8, output_size = 24x24x12, RF = 5x5

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        ) # input_size = 24x24x12, output_size = 24x24x10, RF = 5x5
        
        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # input_size = 24x24x10, output_size = 12x12x10, RF = 6x6
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            #nn.Dropout(dropout_value)
        ) # input_size = 12x12x10, output_size = 10x10x14, RF = 10x10
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            #nn.Dropout(dropout_value)
        ) # input_size = 10x10x14, output_size = 8x8x16, RF = 14x14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            #nn.Dropout(dropout_value)
        ) # input_size = 8x8x16, output_size = 6x6x20, RF = 18x18

        self.pool2 = nn.MaxPool2d(2, 2) # input_size = 6x620, output_size = 3x3x20, RF = 6x6
        # OUTPUT BLOCK
        #self.gap = nn.Sequential(
        #    nn.AvgPool2d(kernel_size=6)
        #) # input_size = 6x6x20, output_size = 1x1x20, RF = 28x28

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(3, 3), padding=0, bias=False)
        ) # input_size = 3x3x20, output_size = 1x1x16, RF = 28x28

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x16, output_size = 1x1x10, RF = 28x28

        #self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        #print ("Input ", x.shape)
        x = self.convblock1(x)
        #print ("Conv1", x.shape)
        x = self.convblock2(x)
        #print ("Conv2", x.shape)
        x = self.convblock3(x)
        #print ("Conv3", x.shape)
        x = self.pool1(x)
        #print ("Pool", x.shape)
        x = self.convblock4(x)
        #print ("Conv4", x.shape)
        x = self.convblock5(x)
        #print ("Conv5", x.shape)
        x = self.convblock6(x)
        x = self.pool2(x)
        #print ("Conv6", x.shape)
        x = self.convblock7(x)
        #print ("Conv7", x.shape)
        x = self.convblock8(x)
        #print ("Conv8", x.shape)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)
    

'''

General Formula for Convolution:
RFout=RFin+(kernel size−1)×j
RFout​ =RFin +(kernel size−1)×j
𝑗out=𝑗in×stride
jout​ =jin​ ×stride

General Formula for Pooling:
RFout=RFin+(pool size−1)×𝑗in
RFout​ =RFin​ +(pool size−1)×jin
​ 
𝑗out=𝑗in×stride
jout​ =jin​ ×stride

Where 𝑗
j is the jump size (which is 1 for stride 1 convolution and increases by the stride value for pooling).

Initial Values:
Input size: 28x28
Receptive field (RF): 1
Jump size (j): 1
Layer-by-Layer Calculation:
Convblock1:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 1 + (3 - 1) * 1 = 3
Convblock2:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 3 + (3 - 1) * 1 = 5
Convblock3:

Kernel size: 1x1, Stride: 1, Padding: 0
RF = 5 + (1 - 1) * 1 = 5
MaxPool2d (pool1):

Kernel size: 2x2, Stride: 2
RF = 5 + (2 - 1) * 1 = 6
j = 1 * 2 = 2 (jump size doubles due to stride 2)
Convblock4:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 6 + (3 - 1) * 2 = 10
Convblock5:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 10 + (3 - 1) * 2 = 14
Convblock6:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 14 + (3 - 1) * 2 = 18
MaxPool2d (pool2):

Kernel size: 2x2, Stride: 2
RF = 18 + (2 - 1) * 2 = 20
j = 2 * 2 = 4 (jump size doubles again due to stride 2)
Convblock7:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 20 + (3 - 1) * 4 = 28
Convblock8:

Kernel size: 1x1, Stride: 1, Padding: 0
RF = 28 + (1 - 1) * 4 = 28
Summary of Receptive Field after Each Block:
Convblock1: RF = 3
Convblock2: RF = 5
Convblock3: RF = 5
MaxPool2d (pool1): RF = 6
Convblock4: RF = 10
Convblock5: RF = 14
Convblock6: RF = 18
MaxPool2d (pool2): RF = 20
Convblock7: RF = 28
Convblock8: RF = 28

'''

class Model_1_6(nn.Module):

    def __init__(self, dropout_value=0):
        super(Model_1_6, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # input_size = 28x28x1, output_size = 26x26x8, RF = 3x3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # input_size = 26x26x8, output_size = 24x24x12, RF = 5x5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size = 24x24x12, output_size = 24x24x10, RF = 5x5
        self.pool1 = nn.MaxPool2d(2, 2) # input_size = 24x24x10, output_size = 12x12x10, RF = 6x6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        ) # input_size = 12x12x10, output_size = 10x10x14, RF = 8x8
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # input_size = 10x10x14, output_size = 8x8x16, RF = 10x10
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value)
        ) # input_size = 8x8x16, output_size = 6x6x20, RF = 12x12

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # input_size = 6x6x20, output_size = 1x1x20, RF = 24x24

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x20, output_size = 1x1x16, RF = 26x2

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x16, output_size = 1x1x10, RF = 28x28

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        #print ("Input ", x.shape)
        x = self.convblock1(x)
        #print ("Conv1", x.shape)
        x = self.convblock2(x)
        #print ("Conv2", x.shape)
        x = self.convblock3(x)
        #print ("Conv3", x.shape)
        x = self.pool1(x)
        #print ("Pool", x.shape)
        x = self.convblock4(x)
        #print ("Conv4", x.shape)
        x = self.convblock5(x)
        #print ("Conv5", x.shape)
        x = self.convblock6(x)
        #print ("Conv6", x.shape)
        x = self.gap(x)
        #print ("Gap", x.shape)
        x = self.convblock7(x)
        #print ("Conv7", x.shape)
        x = self.convblock8(x)
        #print ("Conv8", x.shape)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)
    
    '''
    Initial Values:
Input size: 28x28
Receptive field (RF): 1
Jump size (j): 1
Layer-by-Layer Calculation:
Convblock1:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 1 + (3 - 1) * 1 = 3
𝑗
j = 1
Convblock2:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 3 + (3 - 1) * 1 = 5
𝑗
j = 1
Convblock3:

Kernel size: 1x1, Stride: 1, Padding: 0
RF = 5 + (1 - 1) * 1 = 5
𝑗
j = 1
MaxPool2d (pool1):

Kernel size: 2x2, Stride: 2
RF = 5 + (2 - 1) * 1 = 6
𝑗
j = 1 * 2 = 2
Convblock4:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 6 + (3 - 1) * 2 = 10
𝑗
j = 2
Convblock5:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 10 + (3 - 1) * 2 = 14
𝑗
j = 2
Convblock6:

Kernel size: 3x3, Stride: 1, Padding: 0
RF = 14 + (3 - 1) * 2 = 18
𝑗
j = 2
AvgPool2d (gap):

Kernel size: 6x6, Stride: 1
RF = 18 + (6 - 1) * 2 = 28
𝑗
j = 2
Convblock7:

Kernel size: 1x1, Stride: 1, Padding: 0
RF = 28 + (1 - 1) * 2 = 28
𝑗
j = 2
Convblock8:

Kernel size: 1x1, Stride: 1, Padding: 0
RF = 28 + (1 - 1) * 2 = 28
𝑗
j = 2
Summary of Receptive Field after Each Block:
Convblock1: RF = 3
Convblock2: RF = 5
Convblock3: RF = 5
MaxPool2d (pool1): RF = 6
Convblock4: RF = 10
Convblock5: RF = 14
Convblock6: RF = 18
AvgPool2d (gap): RF = 28
Convblock7: RF = 28
Convblock8: RF = 28
    '''