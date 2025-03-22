import torch
import torch.nn as nn

class CustomSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CustomSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
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
    def __init__(self, in_channels, out_channels):
        super(ChannelPad, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if in_channels > 0:
            spacing = out_channels / in_channels
            self.indices = [int(i * spacing) for i in range(in_channels)]
        else:
            self.indices = []
    
    def forward(self, x):
        b, _, h, w = x.size()
        result = torch.zeros(b, self.out_channels, h, w, device=x.device)
        
        for i, idx in enumerate(self.indices):
            result[:, idx] = x[:, i]
        
        return result


class AggTensors(nn.Module):
            def __init__(self, aggregation = 'sum'):
                super(AggTensors, self).__init__()
                if aggregation != 'sum':
                    raise NotImplementedError(f"Aggregation type {aggregation} not implemented, only 'sum' is supported for now")
                self.aggregation = aggregation

            def forward(self, stacked_x: list):
                return stacked_x.sum(dim=0)