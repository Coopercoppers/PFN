import torch
import torch.nn as nn

class Unsqueezer(nn.Module):
    def __init__(self):
        super(Unsqueezer, self).__init__()
        
    def forward(self, x):
        return torch.unsqueeze(x,1)
    
class Transposer(nn.Module):
    def __init__(self):
        super(Transposer, self).__init__()

    def forward(self, x):
        return torch.transpose(x, -1, -2)
    

class ReshapeCNN(nn.Module):
    def __init__(self):
        super(ReshapeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)  # Convolution to downsample spatial dimensions
        self.deconv = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Transposed convolution to upsample spatial dimensions
        
        self.linear_layer1 = torch.nn.Linear(768, 96)
        self.linearlayer2 = torch.nn.Linear(100,96)
        self.transposer = Transposer()
        self.unsqueezer = Unsqueezer()

    def forward(self, x):
        x = self.unsqueezer(x)
        # Apply convolution to downsample spatial dimensions
        x = self.conv1(x)

        # Apply transposed convolution to upsample spatial dimensions
        x = self.deconv(x)
        
        x = self.linear_layer1(x)

        x=self.transposer(x)

        x = self.linearlayer2(x)

        reducer = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        x = reducer(x)
        return x

tensor = torch.randn((16, 100, 768))

tensor = tensor.transpose(0,2)
print(tensor.shape)