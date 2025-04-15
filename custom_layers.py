import torch
import torch.nn as nn

class CustomSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CustomSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    

class ChannelPad(nn.Module):
    """
    A module that expands the channel dimension of a tensor using different padding modes.
    
    Padding modes:
    - 'zero': Pads with zeros at the end of channels
    - 'repeat': Repeats the input channels (requires out_channels to be divisible by in_channels)
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        padding_mode (str): Padding mode, either 'zero' or 'repeat'
    """
    def __init__(self, in_channels, out_channels, padding_mode='zero'):
        super(ChannelPad, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Validate padding mode
        valid_modes = ['zero', 'repeat']
        if padding_mode not in valid_modes:
            raise ValueError(f"padding_mode must be one of {valid_modes}, got {padding_mode}")
        self.padding_mode = padding_mode
        
        # Check that input channels is not zero
        if in_channels <= 0:
            raise ValueError(f"Input channels ({in_channels}) must be > 0")
            
        # For repeat mode, ensure out_channels is divisible by in_channels
        if padding_mode == 'repeat' and out_channels % in_channels != 0:
            raise ValueError(
                f"For 'repeat' mode, out_channels ({out_channels}) must be "
                f"divisible by in_channels ({in_channels})"
            )
    
    def forward(self, x):
        # If in_channels and out_channels are equal, return input as is
        if self.in_channels == self.out_channels:
            return x
            
        b, c, h, w = x.size()
        
        if self.padding_mode == 'zero':
            padding_channels = self.out_channels - self.in_channels
            padding = torch.zeros(b, padding_channels, h, w, device=x.device)
            result = torch.cat([x, padding], dim=1)

        elif self.padding_mode == 'repeat':
            repeats = self.out_channels // self.in_channels
            result = x.repeat(1, repeats, 1, 1)
        
        return result


class AggTensors(nn.Module):
    def __init__(self, aggregation = 'sum'):
        super(AggTensors, self).__init__()
        if aggregation != 'sum':
            raise NotImplementedError(f"Aggregation type {aggregation} not implemented, only 'sum' is supported for now")
        self.aggregation = aggregation

    def forward(self, stacked_x):
        return stacked_x.sum(dim=0)
            

