import torch.nn as nn
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from spdnet.optimizer import StiefelMetaOptimizer

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 4, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(2)
    )


def conv_block_2d(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class EMGnet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    Model for the features extracted from Riemannain approach
    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(EMGnet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class EMGnet_shallow(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    Model for the features extracted from Riemannain approach
    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(EMGnet_shallow, self).__init__()
        self.encoder = nn.Sequential(
            conv_block_2d(x_dim, hid_dim),
            conv_block_2d(hid_dim, hid_dim),
            conv_block_2d(hid_dim, z_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class EMGnet_raw(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=8, hid_dim=64, z_dim=64):
        super(EMGnet_raw, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def SPD_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        SPDTransform(in_channels, out_channels),
        SPDRectified(),
    )


class SPDnet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self):
        super(SPDnet, self).__init__()

        self.encoder = nn.Sequential(
            SPD_block(8, 8),
            SPD_block(8, 8),
            SPD_block(8, 8),
            SPDTangentSpace(8),
        )
        # self.encoder = nn.Sequential(
        #     SPD_block(hid_dim, hid_dim),
        #     SPD_block(hid_dim, hid_dim),
        #     SPD_block(hid_dim, hid_dim),
        #     SPDTangentSpace(z_dim),
        # )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)